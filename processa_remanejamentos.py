# -*- coding: utf-8 -*-
"""
Remanejamento de estruturas de embalagem com IA/ML

O que este script faz
- Detecta outliers de comportamento (cobertura/consumo/estoque) via IsolationForest (scikit-learn)
  * Recebedoras outliers: processadas por último (baixa prioridade)
  * Doadores outliers: evitados; usados apenas como último recurso
- Abate dinamicamente o estoque do doador a cada remanejamento
- Garante que doadores preservem um "estoque mínimo" (por padrão: >= 60 dias de consumo)
- Remessas em múltiplos de 5
  * arredonda para cima
  * fallback para o maior múltiplo de 5 possível (>= mínimo por embalagem)
- Quando need < mínimo: tenta enviar o MÍNIMO (se houver capacidade mantendo estoque mínimo),
  senão marca como NAO_ATENDIDA
- Grava ATENDIDA, PARCIAL e NAO_ATENDIDA (esta última com doador nulo e quantidade=0)
- Idempotência: remove previamente as linhas do DIA na tabela destino antes de inserir

⚠️ Segurança
Este projeto NÃO deve ter credenciais hardcoded. Configure o banco via variáveis de ambiente:
- Preferencialmente: DATABASE_URL (SQLAlchemy)
- Alternativamente: PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

Requisitos:
  pip install -r requirements.txt

Execução:
  python processa_remanejamentos.py
  python processa_remanejamentos.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sklearn.ensemble import IsolationForest

# Carrega .env (opcional)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


_TABLE_RE = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")


@dataclass(frozen=True)
class Settings:
    database_url: str
    lookback_days: int = 60
    cover_target_days: int = 30
    donor_keep_days: int = 60
    rupture_threshold: int = 3
    only_prefixes: Sequence[str] = ("AC ", "CE ")
    target_table: str = "public.fato_estoque_remanejamento"
    iforest_estimators: int = 200
    iforest_contamination: float = 0.05
    iforest_random_state: int = 42
    today: date = date.today()
    log_level: str = "INFO"
    dry_run: bool = False


# =========================
# Config / Segurança
# =========================
def build_database_url() -> str:
    """
    Resolve a conexão com o Postgres.
    Prioridade:
      1) DATABASE_URL (SQLAlchemy URL)
      2) PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASS
    """
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        return url

    host = (os.getenv("PG_HOST") or "").strip()
    port = (os.getenv("PG_PORT") or "5432").strip()
    db = (os.getenv("PG_DB") or "").strip()
    user = (os.getenv("PG_USER") or "").strip()
    pwd = (os.getenv("PG_PASS") or "").strip()

    missing = [k for k, v in [("PG_HOST", host), ("PG_DB", db), ("PG_USER", user), ("PG_PASS", pwd)] if not v]
    if missing:
        raise ValueError(
            "Configuração de banco ausente. Defina DATABASE_URL ou as variáveis: "
            + ", ".join(missing)
        )

    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def validate_table_name(schema_table: str) -> str:
    """
    Protege contra SQL injection via nome de tabela (não parametrizável em SQLAlchemy text()).
    Exige o formato: schema.table, com caracteres [A-Za-z0-9_].
    """
    if not _TABLE_RE.match(schema_table):
        raise ValueError(
            f"Nome de tabela inválido: {schema_table!r}. Use o formato schema.table (ex: public.minha_tabela)."
        )
    return schema_table


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# =========================
# Utilidades de regra de negócio
# =========================
def is_own_unit(unidade: str, prefixes: Sequence[str]) -> bool:
    return isinstance(unidade, str) and unidade.startswith(tuple(prefixes))


def min_shipment(estrutura_embalagem: str) -> int:
    s = (estrutura_embalagem or "").upper()
    if "CAIXA ENCOMENDA P" in s:
        return 10
    if "ENVELOPE" in s:
        return 10
    if "CAIXA ENCOMENDA M" in s:
        return 5
    if "CAIXA ENCOMENDA G" in s:
        return 5
    return 10


def ceil_to_5(x: int) -> int:
    x = int(x)
    return x if x % 5 == 0 else x + (5 - x % 5)


def floor_to_5(x: int) -> int:
    x = int(x)
    return x - (x % 5)


def safe_days_cover(estoque: Any, consumo: Any) -> float:
    """
    Cobertura em dias: estoque/consumo.
    - consumo <= 0 => inf (sem consumo aparente)
    - estoque nulo => 0
    """
    try:
        if consumo is None or pd.isna(consumo) or float(consumo) <= 0:
            return np.inf
        if estoque is None or pd.isna(estoque):
            return 0.0
        return float(estoque) / float(consumo)
    except Exception:
        return np.inf


def compute_need(estoque: Any, consumo: Any, target_days: int) -> int:
    """
    Necessidade para recompor 'target_days' dias: ceil(target_days*consumo - estoque).
    """
    if consumo is None or pd.isna(consumo):
        return 0
    c = float(consumo)
    if c <= 0:
        return 0
    e = 0.0 if (estoque is None or pd.isna(estoque)) else float(estoque)
    return max(0, int(math.ceil(target_days * c - e)))


# =========================
# Estado dinâmico dos doadores
# =========================
def build_donors_state(snapshot: pd.DataFrame, donor_keep_days: int) -> pd.DataFrame:
    """
    Estado por (gestor, unidade, idmcu, estrutura):
      consumo, estoque_curr, keep(=ceil(donor_keep_days*consumo)), avail, iforest_flag
    """
    df = snapshot[
        [
            "gestor",
            "unidade",
            "idmcu",
            "estrutura_codigo",
            "estrutura_embalagem",
            "consumo_filled",
            "estoque",
            "iforest_flag",
        ]
    ].copy()

    df.rename(columns={"consumo_filled": "consumo"}, inplace=True)
    df["consumo"] = pd.to_numeric(df["consumo"], errors="coerce").fillna(0.0)
    df["estoque_curr"] = pd.to_numeric(df["estoque"], errors="coerce").fillna(0.0)
    df["keep"] = np.ceil(donor_keep_days * df["consumo"])  # manter X dias
    df["avail"] = np.floor(df["estoque_curr"] - df["keep"]).clip(lower=0).astype(int)
    return df


def current_avail(row: pd.Series) -> int:
    return int(max(0, math.floor(float(row["estoque_curr"]) - float(row["keep"]))))


def allocate_from_donors_state(
    donors_state: pd.DataFrame,
    gestor: str,
    estrutura_codigo: int,
    recebedora_unidade: str,
    target_total: int,
    per_donor_min: int = 1,
) -> List[Dict[str, Any]]:
    """
    Aloca até 'target_total', atualizando donors_state (abatimento dinâmico).

    Priorização:
      1) Doadores INLIER (iforest_flag == 1)
      2) Maior disponibilidade (avail)
      3) Maior cobertura atual (days_cover_curr)

    Retorna:
      [{'donor_unidade': str, 'donor_idmcu': int|None, 'qtd': int}, ...]
    """
    allocations: List[Dict[str, Any]] = []
    remaining = int(target_total)
    if remaining <= 0:
        return allocations

    base_mask = (
        (donors_state["gestor"] == gestor)
        & (donors_state["estrutura_codigo"] == estrutura_codigo)
        & (donors_state["unidade"] != recebedora_unidade)
    )

    while remaining > 0:
        candidates = donors_state[base_mask].copy()
        if candidates.empty:
            break

        # Avail com regra de estoque mínimo
        candidates["avail"] = candidates.apply(current_avail, axis=1)
        candidates = candidates[candidates["avail"] >= per_donor_min]
        if candidates.empty:
            break

        # IA/ML: inliers primeiro
        candidates["ml_inlier"] = (candidates["iforest_flag"] == 1).astype(int)
        candidates["days_cover_curr"] = candidates.apply(
            lambda r: safe_days_cover(r["estoque_curr"], r["consumo"]), axis=1
        )

        candidates = candidates.sort_values(
            by=["ml_inlier", "avail", "days_cover_curr"], ascending=[False, False, False]
        )

        idx = candidates.index[0]
        avail = int(candidates.loc[idx, "avail"])

        give = min(avail, remaining)
        if give < per_donor_min:
            break

        allocations.append(
            {
                "donor_unidade": donors_state.loc[idx, "unidade"],
                "donor_idmcu": int(donors_state.loc[idx, "idmcu"])
                if pd.notna(donors_state.loc[idx, "idmcu"])
                else None,
                "qtd": int(give),
            }
        )

        # Abater estoque do doador
        donors_state.at[idx, "estoque_curr"] = float(donors_state.loc[idx, "estoque_curr"]) - give
        donors_state.at[idx, "avail"] = current_avail(donors_state.loc[idx])
        remaining -= give

    return allocations


# =========================
# Banco / IO
# =========================
def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def read_recent_snapshot(conn: Connection, lookback_days: int, prefixes: Sequence[str]) -> pd.DataFrame:
    """
    Lê a janela recente e devolve o snapshot (último registro por chave).
    """
    if not prefixes:
        raise ValueError("prefixes não pode ser vazio.")

    # Monta filtro de prefixos por parâmetros
    prefix_conds = []
    params: Dict[str, Any] = {"lookback": lookback_days}
    for i, p in enumerate(prefixes):
        key = f"p{i}"
        params[key] = f"{p}%"
        prefix_conds.append(f'unidade LIKE :{key}')

    recent_sql = text(
        f"""
        SELECT
          se, municipio, area, gestor, sto, unidade, "data", mes, mes_num, ano,
          grupo, subgrupo, linha, status, embalagem, embalagem_altura, embalagem_largura,
          embalagem_comprimento, embalagem_formato, estoque, consumo, valor, estrutura_codigo,
          estrutura_embalagem, id_codigo, servico, segmento, familia, idmcu, idsto
        FROM public.fato_embalagens
        WHERE "data" >= (CURRENT_DATE - (INTERVAL '1 day' * :lookback))
          AND ({' OR '.join(prefix_conds)})
        """
    )

    df_recent = pd.read_sql(recent_sql, conn, params=params, parse_dates=["data"])
    if df_recent.empty:
        return df_recent

    # Normalizações numéricas
    for col in ["estoque", "consumo", "estrutura_codigo", "idmcu", "idsto", "valor"]:
        if col in df_recent.columns:
            df_recent[col] = pd.to_numeric(df_recent[col], errors="coerce")

    # Snapshot mais recente por chave
    df_recent = df_recent.sort_values(
        by=["gestor", "unidade", "idmcu", "estrutura_codigo", "estrutura_embalagem", "data"]
    )
    keys = ["gestor", "unidade", "idmcu", "estrutura_codigo", "estrutura_embalagem"]
    snapshot = df_recent.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    # Imputação de consumo por mediana do (gestor, estrutura_codigo)
    grp_median = (
        df_recent.groupby(["gestor", "estrutura_codigo"])["consumo"]
        .median()
        .rename("cons_median")
        .reset_index()
    )
    snapshot = snapshot.merge(grp_median, on=["gestor", "estrutura_codigo"], how="left")
    snapshot["consumo_filled"] = snapshot["consumo"].fillna(snapshot["cons_median"])
    snapshot.drop(columns=["cons_median"], inplace=True)
    snapshot["estoque"] = snapshot["estoque"].fillna(0)

    # Cobertura
    snapshot["days_cover"] = snapshot.apply(
        lambda r: safe_days_cover(r["estoque"], r["consumo_filled"]), axis=1
    )

    return snapshot


def apply_isolation_forest(
    snapshot: pd.DataFrame,
    n_estimators: int,
    contamination: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Adiciona coluna iforest_flag (+1 inlier / -1 outlier).
    Em caso de falha, marca todos como inliers.
    """
    snap = snapshot.copy()
    try:
        dc = snap["days_cover"].replace([np.inf, -np.inf], np.nan)
        dc_filled = dc.fillna(np.nanmedian(dc))

        feat = pd.DataFrame(
            {
                "log_dc": np.log1p(dc_filled.astype(float)),
                "log_est": np.log1p(snap["estoque"].fillna(0).astype(float)),
                "log_cons": np.log1p(snap["consumo_filled"].clip(lower=0).fillna(0).astype(float)),
            }
        )

        iso = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=random_state
        )
        preds = iso.fit_predict(feat.values)
        snap["iforest_flag"] = preds
        logging.info(
            "IA/ML IsolationForest: inliers=%d | outliers=%d",
            int((preds == 1).sum()),
            int((preds == -1).sum()),
        )
    except Exception as e:
        logging.warning("Falha no IsolationForest; marcando todos como inliers. Erro: %s", e)
        snap["iforest_flag"] = 1
    return snap


def ensure_target_table(conn: Connection, target_table: str) -> None:
    """
    Cria tabela de destino (se necessário). O nome da tabela já deve estar validado.
    """
    conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                gestor TEXT,
                data DATE NOT NULL,
                recebedora_mcu BIGINT,
                recebedora_unidade TEXT,
                estrutura_codigo BIGINT,
                estrutura_embalagem TEXT,
                distribuidora_mcu BIGINT,
                distribuidora_unidade TEXT,
                quantidade_total INTEGER,
                quantidade INTEGER,
                unidade_status TEXT
            )
            """
        )
    )

    # Compatibilidade (caso exista tabela antiga sem coluna)
    conn.execute(text(f"ALTER TABLE {target_table} ADD COLUMN IF NOT EXISTS unidade_status TEXT"))


def delete_day(conn: Connection, target_table: str, day: date) -> int:
    res = conn.execute(text(f"DELETE FROM {target_table} WHERE data = :d"), {"d": day})
    return int(res.rowcount or 0)


def insert_rows(conn: Connection, target_table: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    conn.execute(
        text(
            f"""
            INSERT INTO {target_table}
            (gestor, data, recebedora_mcu, recebedora_unidade, estrutura_codigo, estrutura_embalagem,
             distribuidora_mcu, distribuidora_unidade, quantidade_total, quantidade, unidade_status)
            VALUES
            (:gestor, :data, :recebedora_mcu, :recebedora_unidade, :estrutura_codigo, :estrutura_embalagem,
             :distribuidora_mcu, :distribuidora_unidade, :quantidade_total, :quantidade, :unidade_status)
            """
        ),
        rows,
    )


# =========================
# Pipeline
# =========================
def run(settings: Settings) -> None:
    target_table = validate_table_name(settings.target_table)
    engine = make_engine(settings.database_url)

    with engine.begin() as conn:
        conn.execute(text("SET LOCAL search_path TO public, pg_catalog"))

        db, user = conn.execute(text("SELECT current_database(), current_user")).one()
        sp = conn.execute(text("SHOW search_path")).scalar_one()
        logging.info("Conectado em DB=%s | user=%s | search_path=%s", db, user, sp)

        logging.info("Lendo dados recentes da fato_embalagens (lookback=%dd)...", settings.lookback_days)
        snapshot = read_recent_snapshot(conn, settings.lookback_days, settings.only_prefixes)
        if snapshot.empty:
            logging.warning("Nenhum dado recente encontrado (últimos %d dias).", settings.lookback_days)
            return

        # IA/ML: outliers
        snapshot = apply_isolation_forest(
            snapshot,
            n_estimators=settings.iforest_estimators,
            contamination=settings.iforest_contamination,
            random_state=settings.iforest_random_state,
        )

        # Filtros de negócio
        snapshot = snapshot[snapshot["unidade"].apply(lambda u: is_own_unit(u, settings.only_prefixes))]
        snapshot = snapshot[pd.notna(snapshot["gestor"])]

        # Déficits: cobertura <= threshold e consumo > 0
        deficits = snapshot[
            (snapshot["days_cover"] <= settings.rupture_threshold) & (snapshot["consumo_filled"] > 0)
        ].copy()
        if deficits.empty:
            logging.info("Nenhuma unidade com cobertura ≤ %d dias.", settings.rupture_threshold)
            return

        deficits["needed_total"] = deficits.apply(
            lambda r: compute_need(r["estoque"], r["consumo_filled"], settings.cover_target_days),
            axis=1,
        )
        deficits = deficits[deficits["needed_total"] > 0].copy()
        if deficits.empty:
            logging.info("Nenhuma necessidade positiva para recompor %d dias.", settings.cover_target_days)
            return

        # Estado dinâmico dos doadores (com flag de ML)
        donors_state = build_donors_state(snapshot, donor_keep_days=settings.donor_keep_days)

        # Prioridade: recebedoras inliers primeiro, depois outliers
        deficits["ml_inlier"] = (deficits["iforest_flag"] == 1).astype(int)
        deficits = deficits.sort_values(by=["ml_inlier", "days_cover", "needed_total"], ascending=[False, True, False])

        # Tabela destino e idempotência do dia
        ensure_target_table(conn, target_table)
        deleted = delete_day(conn, target_table, settings.today)
        logging.info("Idempotência: removidas %d linhas de %s com data=%s.", deleted, target_table, settings.today)

        inserts: List[Dict[str, Any]] = []
        atendidas = parcial = nao_atendidas = 0

        for _, rec in deficits.iterrows():
            gestor = rec["gestor"]
            unidade_rec = rec["unidade"]
            idmcu_rec = int(rec["idmcu"]) if pd.notna(rec["idmcu"]) else None
            ecod = int(rec["estrutura_codigo"]) if pd.notna(rec["estrutura_codigo"]) else None
            eemb = rec["estrutura_embalagem"]
            need = int(rec["needed_total"])
            min_total = min_shipment(eemb)

            # Candidatos (mesmo gestor/estrutura, exclui recebedora)
            mask = (
                (donors_state["gestor"] == gestor)
                & (donors_state["estrutura_codigo"] == ecod)
                & (donors_state["unidade"] != unidade_rec)
            )
            candidates = donors_state[mask].copy()
            if candidates.empty:
                inserts.append(
                    {
                        "gestor": gestor,
                        "data": settings.today,
                        "recebedora_mcu": idmcu_rec,
                        "recebedora_unidade": unidade_rec,
                        "estrutura_codigo": ecod,
                        "estrutura_embalagem": eemb,
                        "distribuidora_mcu": None,
                        "distribuidora_unidade": None,
                        "quantidade_total": need,
                        "quantidade": 0,
                        "unidade_status": "NAO_ATENDIDA",
                    }
                )
                nao_atendidas += 1
                continue

            # Disponibilidade total (mantendo estoque mínimo)
            candidates["avail"] = candidates.apply(current_avail, axis=1)
            total_avail_inliers = int(candidates.loc[candidates["iforest_flag"] == 1, "avail"].sum())
            total_avail_all = int(candidates["avail"].sum())

            # Definir target em múltiplo de 5 e respeitando mínimo
            if need < min_total:
                # Só atende se conseguir pelo menos o mínimo
                if total_avail_inliers >= min_total or total_avail_all >= min_total:
                    target = min_total
                else:
                    inserts.append(
                        {
                            "gestor": gestor,
                            "data": settings.today,
                            "recebedora_mcu": idmcu_rec,
                            "recebedora_unidade": unidade_rec,
                            "estrutura_codigo": ecod,
                            "estrutura_embalagem": eemb,
                            "distribuidora_mcu": None,
                            "distribuidora_unidade": None,
                            "quantidade_total": need,
                            "quantidade": 0,
                            "unidade_status": "NAO_ATENDIDA",
                        }
                    )
                    nao_atendidas += 1
                    continue
            else:
                desired = ceil_to_5(need)

                if total_avail_inliers >= desired or total_avail_all >= desired:
                    target = desired
                else:
                    # Fallback: maior múltiplo de 5 possível (>= mínimo)
                    best = max(min_total, floor_to_5(total_avail_inliers))
                    if best < min_total:
                        best = max(min_total, floor_to_5(total_avail_all))
                    if best < min_total:
                        inserts.append(
                            {
                                "gestor": gestor,
                                "data": settings.today,
                                "recebedora_mcu": idmcu_rec,
                                "recebedora_unidade": unidade_rec,
                                "estrutura_codigo": ecod,
                                "estrutura_embalagem": eemb,
                                "distribuidora_mcu": None,
                                "distribuidora_unidade": None,
                                "quantidade_total": need,
                                "quantidade": 0,
                                "unidade_status": "NAO_ATENDIDA",
                            }
                        )
                        nao_atendidas += 1
                        continue
                    target = best

            # Alocar (prioriza inliers internamente)
            allocs = allocate_from_donors_state(
                donors_state=donors_state,
                gestor=gestor,
                estrutura_codigo=ecod,
                recebedora_unidade=unidade_rec,
                target_total=target,
                per_donor_min=1,
            )
            total_alocado = sum(a["qtd"] for a in allocs)

            if need < min_total:
                unidade_status = "ATENDIDA" if total_alocado >= min_total else "NAO_ATENDIDA"
                if unidade_status == "NAO_ATENDIDA":
                    nao_atendidas += 1
                    inserts.append(
                        {
                            "gestor": gestor,
                            "data": settings.today,
                            "recebedora_mcu": idmcu_rec,
                            "recebedora_unidade": unidade_rec,
                            "estrutura_codigo": ecod,
                            "estrutura_embalagem": eemb,
                            "distribuidora_mcu": None,
                            "distribuidora_unidade": None,
                            "quantidade_total": need,
                            "quantidade": 0,
                            "unidade_status": "NAO_ATENDIDA",
                        }
                    )
                    continue
            else:
                if total_alocado == 0:
                    nao_atendidas += 1
                    inserts.append(
                        {
                            "gestor": gestor,
                            "data": settings.today,
                            "recebedora_mcu": idmcu_rec,
                            "recebedora_unidade": unidade_rec,
                            "estrutura_codigo": ecod,
                            "estrutura_embalagem": eemb,
                            "distribuidora_mcu": None,
                            "distribuidora_unidade": None,
                            "quantidade_total": need,
                            "quantidade": 0,
                            "unidade_status": "NAO_ATENDIDA",
                        }
                    )
                    continue
                unidade_status = "ATENDIDA" if total_alocado >= need else "PARCIAL"

            atendidas += int(unidade_status == "ATENDIDA")
            parcial += int(unidade_status == "PARCIAL")

            for a in allocs:
                inserts.append(
                    {
                        "gestor": gestor,
                        "data": settings.today,
                        "recebedora_mcu": idmcu_rec,
                        "recebedora_unidade": unidade_rec,
                        "estrutura_codigo": ecod,
                        "estrutura_embalagem": eemb,
                        "distribuidora_mcu": a["donor_idmcu"],
                        "distribuidora_unidade": a["donor_unidade"],
                        "quantidade_total": need,
                        "quantidade": int(a["qtd"]),
                        "unidade_status": unidade_status,
                    }
                )

        if not inserts:
            logging.info("Nenhuma linha a inserir.")
            return

        if settings.dry_run:
            logging.info("DRY-RUN: nenhuma escrita no banco. Linhas que seriam inseridas: %d", len(inserts))
            return

        insert_rows(conn, target_table, inserts)

        cnt_tx = conn.execute(
            text(f"SELECT COUNT(*) FROM {target_table} WHERE data = :d"), {"d": settings.today}
        ).scalar_one()
        logging.info("Inseridas %d linhas (ATENDIDA=%d | PARCIAL=%d | NAO_ATENDIDA=%d). Total no dia=%d.",
                     len(inserts), atendidas, parcial, nao_atendidas, int(cnt_tx))

    # Pós-commit (checagem simples)
    engine.dispose()
    engine2 = make_engine(settings.database_url)
    with engine2.connect() as conn2:
        cnt_after = conn2.execute(
            text(f"SELECT COUNT(*) FROM {target_table} WHERE data = :d"), {"d": settings.today}
        ).scalar_one()
        logging.info("Após commit: %d linhas com data=%s na tabela destino.", int(cnt_after), settings.today)
        sample = conn2.execute(
            text(
                f"""
                SELECT recebedora_unidade, estrutura_embalagem, distribuidora_unidade,
                       quantidade, quantidade_total, unidade_status
                FROM {target_table}
                WHERE data = :d
                ORDER BY unidade_status DESC, recebedora_unidade
                LIMIT 10
                """
            ),
            {"d": settings.today},
        ).fetchall()
        if sample:
            logging.info("Amostra (10 linhas): %s", sample)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remanejamento de estruturas de embalagem com IA/ML (IsolationForest).")

    p.add_argument("--database-url", default=os.getenv("DATABASE_URL") or "", help="SQLAlchemy URL. Alternativa às variáveis PG_*.")  # noqa: E501
    p.add_argument("--lookback-days", type=int, default=int(os.getenv("LOOKBACK_DAYS", "60")))
    p.add_argument("--cover-target-days", type=int, default=int(os.getenv("COVER_TARGET_DAYS", "30")))
    p.add_argument("--donor-keep-days", type=int, default=int(os.getenv("DONOR_KEEP_DAYS", "60")))
    p.add_argument("--rupture-threshold", type=int, default=int(os.getenv("RUPTURE_THRESHOLD", "3")))
    p.add_argument("--only-prefixes", default=os.getenv("ONLY_PREFIXES", "AC ,CE "), help='Lista separada por vírgula. Ex: "AC ,CE "')
    p.add_argument("--target-table", default=os.getenv("TARGET_TABLE", "public.fato_estoque_remanejamento"))
    p.add_argument("--iforest-estimators", type=int, default=int(os.getenv("IFOREST_ESTIMATORS", "200")))
    p.add_argument("--iforest-contamination", type=float, default=float(os.getenv("IFOREST_CONTAMINATION", "0.05")))
    p.add_argument("--iforest-random-state", type=int, default=int(os.getenv("IFOREST_RANDOM_STATE", "42")))
    p.add_argument("--today", default=os.getenv("TODAY", ""), help="Data no formato YYYY-MM-DD. Default: hoje.")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    p.add_argument("--dry-run", action="store_true", help="Não grava no banco; apenas processa e loga contagens.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Prefixes
    prefixes = [p.strip() for p in str(args.only_prefixes).split(",") if p.strip()]
    if not prefixes:
        prefixes = ["AC ", "CE "]

    # Date
    today_val = date.today()
    if args.today:
        try:
            today_val = date.fromisoformat(args.today)
        except Exception as e:
            raise ValueError("Argumento --today inválido. Use YYYY-MM-DD.") from e

    database_url = (args.database_url or "").strip() or build_database_url()

    settings = Settings(
        database_url=database_url,
        lookback_days=int(args.lookback_days),
        cover_target_days=int(args.cover_target_days),
        donor_keep_days=int(args.donor_keep_days),
        rupture_threshold=int(args.rupture_threshold),
        only_prefixes=tuple(prefixes),
        target_table=str(args.target_table),
        iforest_estimators=int(args.iforest_estimators),
        iforest_contamination=float(args.iforest_contamination),
        iforest_random_state=int(args.iforest_random_state),
        today=today_val,
        log_level=str(args.log_level),
        dry_run=bool(args.dry_run),
    )

    logging.info("Config: target_table=%s | prefixes=%s | dry_run=%s", settings.target_table, settings.only_prefixes, settings.dry_run)
    run(settings)


if __name__ == "__main__":
    main()

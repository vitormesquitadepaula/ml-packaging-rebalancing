# Remanejamento de Estruturas de Embalagem (IA/ML)

Script em Python para sugerir remanejamentos de estoque de embalagens entre unidades, com regras de negócio e apoio de detecção de outliers (IsolationForest).

## Principais regras implementadas

- **Outliers (IsolationForest)**:
  - Recebedoras outliers são processadas por último.
  - Doadores outliers são evitados; entram apenas como “último recurso”.
- **Estoque mínimo do doador**: preserva no mínimo `DONOR_KEEP_DAYS` dias de consumo (default: 60).
- **Recomposição do recebedor**: calcula necessidade para recompor `COVER_TARGET_DAYS` dias (default: 30).
- **Ruptura**: considera déficit quando `days_cover <= RUPTURE_THRESHOLD` (default: 3).
- **Múltiplos de 5** e **mínimos por embalagem** (ex: envelope/caixa P = 10; caixa M/G = 5).
- **Idempotência diária**: remove previamente as linhas do dia (`data = TODAY`) na tabela de destino antes de inserir.

> A lógica original do algoritmo foi preservada; a principal mudança é **profissionalização de execução e segurança**.

## Segurança (não subir senha para o GitHub)

O script original tinha credenciais hardcoded no código. Isso não pode ir para repositório público/compartilhado.

Use variáveis de ambiente:

### Opção A (recomendada): `DATABASE_URL`

Exemplo:

```
DATABASE_URL=postgresql+psycopg2://USER:SENHA@HOST:5432/DB
```

### Opção B: variáveis separadas

```
PG_HOST=...
PG_PORT=5432
PG_DB=postgres
PG_USER=...
PG_PASS=...
```

Crie um arquivo `.env` local (não versionado) e rode normalmente. Existe um `.env.example` no repositório.

## Como rodar

### 1) Criar venv e instalar dependências

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configurar `.env`

Copie:

```
copy .env.example .env
```

Edite o `.env` com seus dados.

### 3) Executar

```
python processa_remanejamentos.py
```

Dry-run (não grava no banco):

```
python processa_remanejamentos.py --dry-run
```

## Argumentos úteis

- `--lookback-days 60`
- `--cover-target-days 30`
- `--donor-keep-days 60`
- `--rupture-threshold 3`
- `--only-prefixes "AC ,CE "`
- `--target-table public.fato_estoque_remanejamento`
- `--today 2025-12-23`
- `--log-level INFO`
- `--dry-run`

## Estrutura do repositório

- `processa_remanejamentos.py` — script principal
- `requirements.txt` — dependências
- `.env.example` — exemplo de variáveis
- `.gitignore` — evita versionar `.env`, venv, caches, etc.

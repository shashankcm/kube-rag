# Data Ingestion Commands

## Here are the ingestion commands for the 3 useful cases.

### 1. Ingest only clean data

```bash
python -m app.ingestion.processor DATA/true_data true
```

This wipes Qdrant first, then indexes only the clean documents from DATA/true_data.

### 2. Ingest only 10 noisy files

```bash
python -m app.ingestion.processor DATA/noisy_sample_10 noisy
```

### 3. Ingest only 15 noisy files

```bash
python -m app.ingestion.processor DATA/noisy_sample_15 noisy
```

Use --wipe when you want a fresh Qdrant collection. If you want to append noisy files after clean data, omit --wipe on the noisy ingestion command.

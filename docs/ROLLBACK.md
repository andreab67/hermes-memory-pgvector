# Rollback Instructions

If you need to roll back or completely remove Hexus from your environment, follow the steps below.

### 1. Disable the Plugin in Hermes

```bash
hermes config set memory.provider none
sudo systemctl restart hermes.service
```

### 2. (Optional) Drop the Database Tables
> **⚠️ WARNING:** This will cause irreversible data loss.

```bash
sudo -u postgres psql -d <your-memory-db> -c "
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS memory_entries;
"
```

### 3. (Optional) Remove the Plugin Files (if installed manually)

```bash
rm -rf ~/.hermes/plugins/hexus
```

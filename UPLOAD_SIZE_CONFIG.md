# Video Upload Size Configuration

## Current Setup

The TacticEYE2 application now supports **unlimited video file uploads** (configurable).

### Default Configuration

- **Default Max Upload Size**: 500 GB
- **Configurable via Environment Variable**: `MAX_UPLOAD_MB`
- **Upload Processing**: Chunked (1 MB chunks) to minimize memory usage

## How to Configure

### Option 1: Use Default (500 GB)
```bash
python app.py
# Max upload: 500 GB
```

### Option 2: Custom Size via Environment Variable
```bash
# Set to 100 GB
export MAX_UPLOAD_MB=102400
python app.py

# Set to 1 TB
export MAX_UPLOAD_MB=1048576
python app.py

# Set to 10 GB (for limited storage)
export MAX_UPLOAD_MB=10240
python app.py
```

### Option 3: Set in .env file
Create a `.env` file in the project root:
```
MAX_UPLOAD_MB=102400  # 100 GB
```

## Examples

### 5 GB Videos
```bash
export MAX_UPLOAD_MB=5120
python app.py
```

### 50 GB Videos
```bash
export MAX_UPLOAD_MB=51200
python app.py
```

### 1 TB Videos
```bash
export MAX_UPLOAD_MB=1048576
python app.py
```

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV
- WebM
- FLV
- TS
- M4V

## Upload Process

1. **Chunked Upload**: Videos are processed in 1 MB chunks to minimize RAM usage
2. **Real-time Size Check**: File size is validated as it uploads
3. **Automatic Cleanup**: Oversized files are automatically deleted

## Performance Notes

- **Chunk Size**: 1 MB (optimized for slow connections)
- **Memory Usage**: Constant ~1 MB (regardless of file size)
- **Network**: Upload speed depends on network bandwidth
  - 10 Mbps → ~100 seconds per GB
  - 100 Mbps → ~10 seconds per GB
  - 1 Gbps → ~1 second per GB

## Recommendations

### For Different Scenarios

| Scenario | Recommended Size | Notes |
|----------|------------------|-------|
| Live streaming | 10-50 GB | Typical match durations |
| Tournament archive | 100-500 GB | Multi-match recordings |
| Server storage limited | 10-50 GB | Adjust to available disk space |
| Research/testing | 5-20 GB | Development workflows |

### Disk Space Considerations

Remember to leave headroom for:
- Processing temporary files (~2x video size during analysis)
- Heatmap outputs
- System requirements

**Recommended Free Disk Space**: 3x maximum upload video size

## Docker Configuration

If running in Docker, set the environment variable during container startup:

```bash
docker run -e MAX_UPLOAD_MB=102400 tacuceye2-app
```

Or in docker-compose.yml:
```yaml
environment:
  - MAX_UPLOAD_MB=102400
```

---

**Status**: ✅ **Unlimited upload size** (configurable, default 500 GB)

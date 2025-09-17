# macOS Launchd Setup for WallStreetBots

This directory contains configuration for running WallStreetBots automatically on macOS using launchd.

## Setup Instructions

### 1. Prerequisites
- Ensure your `.env` file is configured with your Alpaca API keys
- Virtual environment should be set up at `./venv/`
- All dependencies installed

### 2. Test Manual Execution First
```bash
cd /Users/admin/IdeaProjects/workspace/WallStreetBots
python test_env_keys.py  # Verify API keys are loaded
python simple_bot.py    # Test manual execution (Ctrl+C to stop)
```

### 3. Install Launchd Service
```bash
# Copy the plist file to LaunchAgents directory
cp examples/macos/com.wallstreetbots.trading.plist ~/Library/LaunchAgents/

# Load the service (will start automatically)
launchctl load ~/Library/LaunchAgents/com.wallstreetbots.trading.plist

# Check service status
launchctl list | grep wallstreetbots
```

### 4. View Logs
```bash
# View output logs
tail -f /Users/admin/wallstreetbots_trading.log

# View error logs
tail -f /Users/admin/wallstreetbots_error.log
```

### 5. Control the Service
```bash
# Stop the service
launchctl stop com.wallstreetbots.trading

# Start the service
launchctl start com.wallstreetbots.trading

# Unload the service (disable auto-start)
launchctl unload ~/Library/LaunchAgents/com.wallstreetbots.trading.plist
```

## Schedule Configuration

The current configuration runs:
- **Weekdays**: Monday-Friday at 9:25 AM EST (5 minutes before market open)
- **Auto-restart**: Keeps the bot running if it crashes
- **Background**: Runs as a background process

## Important Notes

⚠️ **Security**: Make sure your `.env` file has proper permissions:
```bash
chmod 600 .env
```

⚠️ **Testing**: Always test in paper trading mode first:
```bash
# In your .env file:
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

⚠️ **Monitoring**: Check logs regularly to ensure proper operation.

## Troubleshooting

### Service Not Starting
1. Check file permissions: `ls -la ~/Library/LaunchAgents/com.wallstreetbots.trading.plist`
2. Verify paths in plist file match your installation
3. Check error logs: `tail -f /Users/admin/wallstreetbots_error.log`

### API Connection Issues
1. Verify API keys: `python test_env_keys.py`
2. Check `.env` file exists and has correct keys
3. Ensure paper trading URL for testing

### Python/Django Issues
1. Activate virtual environment manually and test
2. Check Django settings: `echo $DJANGO_SETTINGS_MODULE`
3. Verify all dependencies installed in venv
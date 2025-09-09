# macOS LaunchAgent Setup

This directory contains macOS LaunchAgent configuration files for running WallStreetBots as a background service.

## Files

- `com.wallstreetbots.trading.plist` - LaunchAgent configuration for automated trading

## Setup Instructions

### 1. Customize the plist file

Edit `com.wallstreetbots.trading.plist` and update the following paths:

```xml
<!-- Update these paths to match your installation -->
<string>/Users/YOUR_USERNAME/IdeaProjects/workspace/WallStreetBots/venv/bin/python</string>
<string>/Users/YOUR_USERNAME/IdeaProjects/workspace/WallStreetBots/simple_bot.py</string>
<string>/Users/YOUR_USERNAME/IdeaProjects/workspace/WallStreetBots</string>
```

### 2. Install the LaunchAgent

```bash
# Copy the plist file to LaunchAgents directory
cp com.wallstreetbots.trading.plist ~/Library/LaunchAgents/

# Load the service
launchctl load ~/Library/LaunchAgents/com.wallstreetbots.trading.plist
```

### 3. Verify the service is running

```bash
# Check if the service is loaded
launchctl list | grep wallstreetbots

# Check the logs
tail -f ~/wallstreetbots_trading.log
tail -f ~/wallstreetbots_error.log
```

### 4. Control the service

```bash
# Start the service
launchctl start com.wallstreetbots.trading

# Stop the service
launchctl stop com.wallstreetbots.trading

# Unload the service
launchctl unload ~/Library/LaunchAgents/com.wallstreetbots.trading.plist
```

## Configuration Details

### Trading Schedule
- **Days**: Monday-Friday (Weekdays 1-5)
- **Time**: 9:25 AM (5 minutes before market open)
- **Timezone**: System default

### Logging
- **Output**: `~/wallstreetbots_trading.log`
- **Errors**: `~/wallstreetbots_error.log`

### Service Behavior
- **RunAtLoad**: Starts automatically when system boots
- **KeepAlive**: Restarts automatically if it crashes
- **ExitTimeOut**: 30 seconds to gracefully shutdown

## Security Notes

- The service runs with your user permissions
- API keys are loaded from your `.env` file
- Logs are stored in your home directory
- No root privileges required

## Troubleshooting

### Service won't start
1. Check the error log: `cat ~/wallstreetbots_error.log`
2. Verify paths in the plist file
3. Ensure `.env` file exists with valid API keys
4. Test manually: `python simple_bot.py`

### Service stops unexpectedly
1. Check the trading log: `tail -f ~/wallstreetbots_trading.log`
2. Verify API keys are valid
3. Check network connectivity
4. Review account status in Alpaca dashboard

### Market hours issues
- The service starts at 9:25 AM but trading depends on market hours
- Ensure your bot handles market closed conditions
- Consider adding market hours detection to your bot




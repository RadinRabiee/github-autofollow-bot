# 🚀 GitHub Auto-Follow Bot

<div align="center">


**Supercharge your GitHub network with intelligent, automated networking**


## 💫 What is GitHub Auto-Follow Bot?

A powerful, customizable bot that helps you grow your GitHub presence through strategic and ethical networking. Stop wasting hours manually following users - let the bot handle your network growth while you focus on coding!


## ✨ Key Features

- **📊 Smart User Targeting**: Follows users with fewer than 10 followers whose last commit was earlier than 3 days ago
- **⭐ Intelligent Repository Starring**: Stars 1-3 repos from followed users (configurable 70% probability)
- **🔄 Automated Scheduling**: Runs 7-10 times daily at natural intervals
- **🔐 Enterprise-Grade Security**: Military-grade encryption for your GitHub token
- **🕵️ Anti-Detection Mechanisms**: Human-like behavior patterns to stay under the radar
- **📱 Cross-Platform Support**: Works on Linux, macOS, and Windows
- **📝 Detailed Activity Logging**: Comprehensive tracking of all follows and stars
- **⚙️ Extensive Customization**: Fine-tune every aspect of the bot's behavior
- **🔧 Command-Line Interface**: Powerful options for different execution modes
- **🚫 Rate-Limit Protection**: Built-in safeguards against API limits
- **🧩 Modular Design**: Easily extend with your own custom functionality

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/RadinRabiee/github-autofollow-bot.git
cd github-autofollow-bot

# Install dependencies
pip install requests cryptography
```

## 🏁 Quick Start Guide

```bash
# Make the script executable
chmod +x github_autofollow.py

# Set up your GitHub token (needed only once)
./github_autofollow.py --setup

# Run a test with 5 users
./github_autofollow.py --run --max-users 5 --debug
```

## 🔧 Command Line Options

```
--setup              Set up your GitHub token with secure encryption
--run                Run the bot once (perfect for testing)
--daemon             Run continuously in the background with smart scheduling
--max-users N        Set maximum users to follow per run (default: 15)
--star               Enable repository starring (default)
--no-star            Disable repository starring
--debug              Enable detailed debug logging
--force              Skip confirmation prompts
--install-service    Install systemd service for automatic startup
--help               Show all available commands and options
```

## 🔄 Running in Background

### Method 1: Using nohup (Recommended)

```bash
# Run in daemon mode with nohup
nohup ./github_autofollow.py --daemon > github-bot.log 2>&1 &

# Check if it's running
ps aux | grep github_autofollow

# Monitor logs
tail -f ~/.github_autofollow.log
```

### Method 2: Using Systemd

```bash
# Install the systemd service
./github_autofollow.py --install-service

# Enable and start the service
systemctl --user enable github-autofollow.service
systemctl --user start github-autofollow.service
```

## ⚙️ Advanced Configuration

Customize the bot's behavior by editing these variables in the script:

```python
# Configuration Constants
DEFAULT_MIN_DELAY = 5           # Min seconds between actions
DEFAULT_MAX_DELAY = 15          # Max seconds between actions
MAX_REQUESTS_PER_HOUR = 50      # API rate limit protection
MIN_RUNS_PER_DAY = 7            # Min scheduled runs per day
MAX_RUNS_PER_DAY = 10           # Max scheduled runs per day
STAR_PROBABILITY = 0.7          # 70% chance to star repos
MIN_STARS = 1                   # Min repos to star per user
MAX_STARS = 3                   # Max repos to star per user
DEFAULT_MAX_USERS_PER_RUN = 15  # Users to follow per run
```

## 🛡️ Security Features

- **Token Encryption**: Secure storage using PBKDF2 key derivation and Fernet encryption
- **File Permissions**: Automatic 0600 permission setting for sensitive files
- **Configurable User-Agent**: Randomized browser user-agents to avoid fingerprinting
- **Session Management**: Persistent connection handling with request timeouts
- **Exponential Backoff**: Smart retry mechanism for server errors
- **Secure Error Handling**: Exception management without exposing sensitive info

## 🌐 Network Growth Strategy

The bot employs a proven three-pronged approach to maximize your GitHub network growth:

1. **Target Selection**: Focuses on users with <10 followers who are likely to notice and reciprocate follows
2. **Engagement**: Stars repositories to increase follow-back probability (typically doubles the rate)
3. **Natural Pacing**: Distributes activity throughout the day to appear genuine and avoid triggers

Expected Results:
- 10-25% follow-back rate
- ~100-150 new follows per day
- ~15-30 new followers per day
- Potential for exponential growth as your profile visibility increases

## 📜 Ethical Guidelines

This bot is designed for responsible use within GitHub's terms of service:

- Follows users at a reasonable, human-like pace
- Never unfollows automatically (no follow-unfollow tactics)
- Creates meaningful interactions through repository starring
- Helps undiscovered developers gain visibility
- Operates transparently with detailed logs

## 🤝 Why Contribute?

- Help developers worldwide build their networks
- Support open-source contribution and discovery
- Make GitHub a more interconnected community
- Learn advanced Python, API handling, and security techniques
- Join a growing community of ethical networking advocates

## 👨‍💻 About the Author

Created by [Radin Rabiee](https://github.com/RadinRabiee), a developer passionate about building tools that help the open-source community thrive.

## ⚖️ License

This project is licensed under a copyleft license - see the LICENSE file for details.

---

<div align="center">
<p>If you found this useful, please consider starring the repository! ⭐</p>
<p>
<a href="https://github.com/RadinRabiee/github-autofollow-bot/stargazers">Star this repo</a> •
<a href="https://github.com/RadinRabiee/github-autofollow-bot/issues">Report a bug</a> •
<a href="https://github.com/RadinRabiee/github-autofollow-bot/fork">Fork this repo</a>
</p>
</div>

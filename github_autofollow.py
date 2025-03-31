#!/usr/bin/env python3
"""
GitHub Auto-Follow Bot
----------------------
This script automatically follows GitHub users with specific criteria:
- Users with fewer than 10 followers
- Users whose last commit was earlier than 1 day ago

Additional features:
- Stars 1-3 repositories from followed users (70% probability)
- Runs in the background automatically (5-7 times daily)
- Includes safety features: rate limiting, human-like delays, token encryption
- Logs all activities (follows and stars)
- Command-line options for different execution modes
"""

import argparse
import base64
import configparser
import cryptography
import datetime
import json
import logging
import os
import random
import requests
import signal
import sys
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# Configuration Constants
CONFIG_FILE = os.path.expanduser("~/.github_autofollow.ini")
KEY_FILE = os.path.expanduser("~/.github_autofollow.key")
LOG_FILE = os.path.expanduser("~/.github_autofollow.log")
DEFAULT_MIN_DELAY = 5  # seconds - reduced for testing
DEFAULT_MAX_DELAY = 15  # seconds - reduced for testing
MAX_REQUESTS_PER_HOUR = 50  # GitHub API has rate limits
MIN_RUNS_PER_DAY = 7
MAX_RUNS_PER_DAY = 10
STAR_PROBABILITY = 0.7  # 70% chance to star repos of followed users
MIN_STARS = 1
MAX_STARS = 3
DEFAULT_MAX_USERS_PER_RUN = 15
API_BASE_URL = "https://api.github.com"


# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # This level will be overridden if --debug is specified
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger = logging.getLogger('')  # Root logger
logger.addHandler(console_handler)


class TokenManager:
    """Handles secure storage and retrieval of GitHub API tokens."""
    
    def __init__(self, config_file: str = CONFIG_FILE, key_file: str = KEY_FILE):
        self.config_file = config_file
        self.key_file = key_file
        self.config = configparser.ConfigParser()
        
    def generate_key(self, password: str) -> bytes:
        """Generate a Fernet key from a password."""
        salt = b'github_autofollow_salt_value'  # In production, store this securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
        
    def save_key(self, key: bytes) -> None:
        """Save the encryption key to a file."""
        with open(self.key_file, 'wb') as f:
            f.write(key)
        os.chmod(self.key_file, 0o600)  # Restrict permissions
        
    def load_key(self) -> bytes:
        """Load the encryption key from a file."""
        try:
            with open(self.key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Key file not found: {self.key_file}")
            sys.exit(1)
            
    def encrypt_token(self, token: str, password: str) -> str:
        """Encrypt the GitHub token using the provided password."""
        key = self.generate_key(password)
        fernet = Fernet(key)
        encrypted_token = fernet.encrypt(token.encode())
        self.save_key(key)
        return encrypted_token.decode()
        
    def decrypt_token(self) -> str:
        """Decrypt and return the GitHub token."""
        key = self.load_key()
        self.config.read(self.config_file)
        
        try:
            encrypted_token = self.config.get('github', 'encrypted_token')
            fernet = Fernet(key)
            decrypted_token = fernet.decrypt(encrypted_token.encode())
            return decrypted_token.decode()
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.error(f"Error reading config: {e}")
            sys.exit(1)
        
    def setup_token(self, token: str, password: str) -> None:
        """Set up a new GitHub token with encryption."""
        encrypted_token = self.encrypt_token(token, password)
        
        self.config['github'] = {
            'encrypted_token': encrypted_token,
            'setup_date': datetime.datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        os.chmod(self.config_file, 0o600)  # Restrict permissions
        
        logger.info("GitHub token successfully encrypted and saved.")


class GitHubAPI:
    """Wrapper for GitHub API calls with rate limiting and error handling."""
    
    def __init__(self, token: str):
        self.token = token
        # Using more realistic User-Agent to avoid detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': random.choice(user_agents)
        }
        self.requests_made = 0
        self.last_request_time = 0
        self.session = requests.Session()  # Use a session for better performance and connection reuse
        
    def _delay_request(self, min_delay: int = DEFAULT_MIN_DELAY, max_delay: int = DEFAULT_MAX_DELAY) -> None:
        """Apply a natural delay between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < min_delay:
            # Calculate a random delay within the range
            delay = random.uniform(min_delay - elapsed, max_delay - elapsed)
            logger.debug(f"Waiting {delay:.2f} seconds before next request...")
            time.sleep(delay)
            
        self.last_request_time = time.time()
        
    def _check_rate_limit(self) -> bool:
        """Check if we're about to exceed the rate limit."""
        self.requests_made += 1
        
        if self.requests_made >= MAX_REQUESTS_PER_HOUR:
            logger.warning("Rate limit approaching. Pausing operations.")
            return False
            
        return True
        
    def _make_request(self, method: str, endpoint: str, params: dict = None, data: dict = None, retry_count: int = 0) -> dict:
        """Make an API request with built-in delays and rate limiting."""
        # Apply natural delay
        self._delay_request()
        
        # Check rate limits
        if not self._check_rate_limit():
            # Wait for an hour before resuming
            logger.info("Waiting for 1 hour to respect rate limits...")
            time.sleep(3600)
            self.requests_made = 0
        
        url = f"{API_BASE_URL}{endpoint}"
        logger.debug(f"Making {method.upper()} request to {url}")
        
        try:
            if method.lower() == 'get':
                response = self.session.get(url, headers=self.headers, params=params, timeout=30)
            elif method.lower() == 'post':
                response = self.session.post(url, headers=self.headers, json=data, timeout=30)
            elif method.lower() == 'put':
                response = self.session.put(url, headers=self.headers, json=data, timeout=30)
            elif method.lower() == 'delete':
                response = self.session.delete(url, headers=self.headers, timeout=30)
            else:
                logger.error(f"Unsupported method: {method}")
                return {}
                
            response.raise_for_status()
            logger.debug(f"Request successful: {response.status_code}")
            return response.json() if response.text else {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 403:
                    logger.warning("Rate limit exceeded. Waiting for reset...")
                    # Get rate limit reset time
                    reset_time = int(e.response.headers.get('X-RateLimit-Reset', time.time() + 3600))
                    wait_time = max(0, reset_time - time.time()) + 60  # Add 60 seconds as buffer
                    logger.info(f"Waiting {wait_time:.2f} seconds for rate limit reset...")
                    time.sleep(wait_time)
                    self.requests_made = 0
                    
            return {}
            
    def get_user(self, username: str) -> dict:
        """Get information about a specific user."""
        return self._make_request('get', f'/users/{username}')
        
    def get_user_followers(self, username: str) -> int:
        """Get the number of followers for a user."""
        user_data = self.get_user(username)
        return user_data.get('followers', 0)
        
    def get_user_last_commit(self, username: str) -> Optional[datetime.datetime]:
        """Get the date of the user's last commit."""
        events = self._make_request('get', f'/users/{username}/events')
        
        for event in events:
            if event['type'] in ['PushEvent', 'CreateEvent', 'CommitCommentEvent', 'PullRequestEvent', 'IssuesEvent', 'IssueCommentEvent']:
                commit_date_str = event['created_at']
                return datetime.datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
                
        return None
        
    def search_users(self, query: str, page: int = 1, per_page: int = 30) -> List[dict]:
        """Search for users matching the given criteria."""
        logger.debug(f"Searching users with query: {query}, page: {page}, per_page: {per_page}")
        
        params = {
            'q': query,
            'page': page,
            'per_page': per_page,
            'sort': 'joined',  # Sort by most recently joined to find new users
            'order': 'desc'
        }
        
        response = self._make_request('get', '/search/users', params=params)
        total_count = response.get('total_count', 0)
        items = response.get('items', [])
        logger.debug(f"Search returned {total_count} total users, retrieved {len(items)} items")
        return items
        
    def follow_user(self, username: str) -> bool:
        """Follow a specific user."""
        response = self._make_request('put', f'/user/following/{username}')
        success = response == {}  # GitHub returns empty object on success
        
        if success:
            logger.info(f"Successfully followed user: {username}")
        else:
            logger.error(f"Failed to follow user: {username}")
            
        return success
        
    def get_user_repos(self, username: str, page: int = 1, per_page: int = 100) -> List[dict]:
        """Get repositories for a specific user."""
        params = {
            'page': page,
            'per_page': per_page,
            'sort': 'updated',
            'direction': 'desc'
        }
        
        return self._make_request('get', f'/users/{username}/repos', params=params)
        
    def star_repo(self, owner: str, repo: str) -> bool:
        """Star a specific repository."""
        response = self._make_request('put', f'/user/starred/{owner}/{repo}')
        success = response == {}  # GitHub returns empty object on success
        
        if success:
            logger.info(f"Successfully starred repository: {owner}/{repo}")
        else:
            logger.error(f"Failed to star repository: {owner}/{repo}")
            
        return success


class AutoFollowBot:
    """Main bot class that handles the auto-follow logic."""
    
    def __init__(self, token: str):
        self.api = GitHubAPI(token)
        self.followed_users = set()
        self.starred_repos = set()
        self.load_history()
        
    def load_history(self) -> None:
        """Load history of followed users and starred repos."""
        history_file = os.path.expanduser("~/.github_autofollow_history.json")
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    self.followed_users = set(history.get('followed_users', []))
                    self.starred_repos = set(history.get('starred_repos', []))
                    logger.info(f"Loaded {len(self.followed_users)} followed users and {len(self.starred_repos)} starred repos from history")
            except json.JSONDecodeError:
                logger.error("Failed to load history file, it may be corrupted.")
        else:
            logger.info("No history file found. Starting fresh.")
            
    def save_history(self) -> None:
        """Save history of followed users and starred repos."""
        history_file = os.path.expanduser("~/.github_autofollow_history.json")
        
        with open(history_file, 'w') as f:
            json.dump({
                'followed_users': list(self.followed_users),
                'starred_repos': list(self.starred_repos)
            }, f)
            
        logger.info(f"Saved {len(self.followed_users)} followed users and {len(self.starred_repos)} starred repos to history")
        
    def should_follow_user(self, username: str, user_data: dict = None) -> bool:
        """Check if a user meets the criteria for following."""
        # Skip if already followed
        if username in self.followed_users:
            logger.debug(f"Skipping {username} - already followed")
            return False
            
        logger.debug(f"Checking last commit date for {username}")
        # Check last commit date
        last_commit = self.api.get_user_last_commit(username)
        
        # If no commit history found, we'll still follow them if they're recent users
        if last_commit is None:
            logger.debug(f"No commit history found for {username}, checking account creation date")
            user_info = self.api.get_user(username)
            created_at = user_info.get('created_at')
            if created_at:
                try:
                    account_creation = datetime.datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
                    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
                    if account_creation > thirty_days_ago:
                        logger.debug(f"User {username} has a new account (< 30 days), will follow")
                        return True
                except (ValueError, TypeError):
                    pass
            logger.debug(f"Skipping {username} - no recent activity found")
            return False
            
        three_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        if last_commit > three_days_ago:
            logger.debug(f"Skipping {username} - last commit too recent ({last_commit})")
            return False
            
        logger.debug(f"{username} meets all criteria for following")
        return True
        
    def should_star_repos(self) -> bool:
        """Determine if we should star repos for this user (70% probability)."""
        return random.random() < STAR_PROBABILITY
        
    def star_random_repos(self, username: str) -> None:
        """Star 1-3 random repositories from a user."""
        repos = self.api.get_user_repos(username)
        
        if not repos:
            logger.debug(f"No repositories found for user {username}")
            return
            
        # Filter out already starred repos
        unstarred_repos = [repo for repo in repos if f"{repo['owner']['login']}/{repo['name']}" not in self.starred_repos]
        
        if not unstarred_repos:
            logger.debug(f"No unstarred repositories found for user {username}")
            return
            
        # Determine how many repos to star (1-3)
        num_to_star = min(random.randint(MIN_STARS, MAX_STARS), len(unstarred_repos))
        repos_to_star = random.sample(unstarred_repos, num_to_star)
        
        for repo in repos_to_star:
            owner = repo['owner']['login']
            repo_name = repo['name']
            
            if self.api.star_repo(owner, repo_name):
                self.starred_repos.add(f"{owner}/{repo_name}")
                
        logger.info(f"Starred {num_to_star} repositories for user {username}")
        
    def find_and_follow_users(self, max_users: int = 10) -> int:
        """Find and follow users matching our criteria."""
        # Search for users with fewer than 10 followers
        search_queries = [
            "followers:<10"
        ]
        
        followed_count = 0
        logger.debug("Starting to search for users to follow...")
        
        for query in search_queries:
            if followed_count >= max_users:
                break
                
            logger.debug(f"Searching with query: {query}")
            
            # Check multiple pages if needed (up to 10 pages max)
            max_pages = 10
            current_page = 1
            found_enough_users = False
            
            while current_page <= max_pages and not found_enough_users:
                logger.debug(f"Checking page {current_page} for query: {query}")
                # Get more users per page (maximum allowed by GitHub API)
                users = self.api.search_users(query, page=current_page, per_page=100)
                
                if not users:
                    logger.debug(f"No users found for query on page {current_page}")
                    break
                    
                logger.debug(f"Found {len(users)} potential users for query: {query} (page {current_page})")
                
                # Randomize the order to avoid pattern detection
                random_users = random.sample(users, len(users)) if users else []
                for user in random_users:
                    if followed_count >= max_users:
                        found_enough_users = True
                        break
                        
                    username = user['login']
                    logger.debug(f"Evaluating user: {username}")
                    
                    if self.should_follow_user(username, user):
                        logger.debug(f"User {username} meets criteria, attempting to follow...")
                        if self.api.follow_user(username):
                            self.followed_users.add(username)
                            followed_count += 1
                            logger.info(f"Successfully followed user {followed_count}/{max_users}: {username}")
                            
                            # Possibly star some repositories
                            if self.should_star_repos():
                                self.star_random_repos(username)
                    else:
                        logger.debug(f"User {username} does not meet criteria for following")
                
                current_page += 1
        
        # Save history after each run
        self.save_history()
        return followed_count
        
    def run(self, max_users: int = 10, enable_starring: bool = True) -> int:
        """Run the bot with the specified parameters."""
        logger.info(f"Starting auto-follow bot run (max users: {max_users}, starring: {'enabled' if enable_starring else 'disabled'})")
        
        # If starring is disabled, temporarily modify probability
        original_probability = None
        if not enable_starring:
            original_probability = STAR_PROBABILITY
            globals()['STAR_PROBABILITY'] = 0
            
        try:
            followed = self.find_and_follow_users(max_users)
            logger.info(f"Completed run. Followed {followed} new users.")
            return followed
        finally:
            # Restore original probability
            if not enable_starring and original_probability is not None:
                globals()['STAR_PROBABILITY'] = original_probability


class DaemonManager:
    """Manages the daemon mode of the bot."""
    
    @staticmethod
    def calculate_next_run_time() -> datetime.datetime:
        """Calculate the next run time based on the desired frequency."""
        # Determine how many runs we want today
        runs_today = random.randint(MIN_RUNS_PER_DAY, MAX_RUNS_PER_DAY)
        
        # Calculate the time intervals
        now = datetime.datetime.now()
        start_time = datetime.datetime(now.year, now.month, now.day, 8, 0)  # 8 AM
        end_time = datetime.datetime(now.year, now.month, now.day, 22, 0)   # 10 PM
        
        # If we're past the end time, schedule for tomorrow
        if now > end_time:
            start_time += datetime.timedelta(days=1)
            end_time += datetime.timedelta(days=1)
            
        # If we're before the start time, use the start time
        if now < start_time:
            return start_time
            
        # Calculate time between runs
        active_seconds = (end_time - start_time).total_seconds()
        interval_seconds = active_seconds / (runs_today + 1)
        
        # Add some randomness to the interval
        jitter = random.uniform(-0.1 * interval_seconds, 0.1 * interval_seconds)
        next_run_seconds = interval_seconds + jitter
        
        next_run_time = now + datetime.timedelta(seconds=next_run_seconds)
        
        # Ensure it's not past the end time
        if next_run_time > end_time:
            next_run_time = start_time + datetime.timedelta(days=1)
            
        return next_run_time
        
    @staticmethod
    def run_daemon(args: argparse.Namespace) -> None:
        """Run the bot in daemon mode."""
        logger.info("Starting daemon mode")
        
        # Load token
        token_manager = TokenManager()
        token = token_manager.decrypt_token()
        
        # Create bot instance
        bot = AutoFollowBot(token)
        
        # Set up signal handlers
        def handle_signal(sig, frame):
            logger.info("Received signal to terminate. Shutting down gracefully...")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # Immediately run once when starting daemon mode
        logger.info("Running initial execution before scheduling...")
        try:
            users_followed = bot.run(
                max_users=args.max_users,
                enable_starring=args.star
            )
            logger.info(f"Initial daemon run completed. Followed {users_followed} users.")
        except Exception as e:
            logger.error(f"Error during initial daemon run: {e}")
        
        # Main daemon loop
        try:
            while True:
                next_run = DaemonManager.calculate_next_run_time()
                logger.info(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Sleep until the next run time with jitter to appear more natural
                sleep_seconds = (next_run - datetime.datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    # Add a small random variation (±5 minutes) to avoid exact patterns
                    jitter = random.uniform(-300, 300)  # ±5 minutes in seconds
                    adjusted_sleep = max(0, sleep_seconds + jitter)
                    logger.debug(f"Sleeping for {adjusted_sleep:.2f} seconds (with jitter)")
                    time.sleep(adjusted_sleep)
                    
                # Run the bot
                try:
                    users_followed = bot.run(
                        max_users=args.max_users,
                        enable_starring=args.star
                    )
                    logger.info(f"Daemon run completed. Followed {users_followed} users.")
                except Exception as e:
                    logger.error(f"Error during daemon run: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in daemon mode: {e}")
            
        logger.info("Daemon mode terminated")


def create_systemd_service() -> None:
    """Create a systemd service for running the bot."""
    script_path = os.path.abspath(sys.argv[0])
    service_content = f"""[Unit]
Description=GitHub Auto-Follow Bot
After=network.target

[Service]
ExecStart={sys.executable} {script_path} --daemon
Restart=on-failure
User={os.getenv('USER')}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
    
    service_path = os.path.expanduser("~/.config/systemd/user/github-autofollow.service")
    os.makedirs(os.path.dirname(service_path), exist_ok=True)
    
    with open(service_path, 'w') as f:
        f.write(service_content)
        
    logger.info(f"Created systemd service at {service_path}")
    logger.info("To enable and start the service, run:")
    logger.info("  systemctl --user enable github-autofollow.service")
    logger.info("  systemctl --user start github-autofollow.service")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GitHub Auto-Follow Bot")
    
    parser.add_argument('--setup', action='store_true', help='Set up the GitHub token')
    parser.add_argument('--run', action='store_true', help='Run the bot once')
    parser.add_argument('--daemon', action='store_true', help='Run in daemon mode')
    parser.add_argument('--force', action='store_true', help='Force operation without confirmation')
    parser.add_argument('--max-users', type=int, default=DEFAULT_MAX_USERS_PER_RUN, help='Maximum number of users to follow per run')
    parser.add_argument('--star', action='store_true', default=True, help='Enable repository starring')
    parser.add_argument('--no-star', action='store_false', dest='star', help='Disable repository starring')
    parser.add_argument('--install-service', action='store_true', help='Install systemd service')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_arguments()
    
    # Set up logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        
    # Handle --setup
    if args.setup:
        token_manager = TokenManager()
        token = input("Enter your GitHub Personal Access Token: ")
        password = input("Enter a password to encrypt your token: ")
        token_manager.setup_token(token, password)
        logger.info("Setup complete. Your token has been encrypted and saved.")
        return
        
    # Handle --install-service
    if args.install_service:
        create_systemd_service()
        return
        
    # For all other commands, we need a token
    token_manager = TokenManager()
    try:
        token = token_manager.decrypt_token()
    except Exception as e:
        logger.error(f"Failed to decrypt token: {e}")
        logger.error("Please run with --setup first to configure your GitHub token.")
        return
        
    # Handle --run
    if args.run:
        bot = AutoFollowBot(token)
        users_followed = bot.run(max_users=args.max_users, enable_starring=args.star)
        logger.info(f"Run completed. Followed {users_followed} users.")
        return
        
    # Handle --daemon
    if args.daemon:
        DaemonManager.run_daemon(args)
        return
        
    # If no command specified, show help
    if not any([args.setup, args.run, args.daemon, args.install_service]):
        logger.info("No command specified. Use --help to see available commands.")
        logger.info("Common usage:")
        logger.info("  Initial setup:  --setup")
        logger.info("  Single run:     --run [--max-users N] [--no-star]")
        logger.info("  Daemon mode:    --daemon [--max-users N] [--no-star]")
        logger.info("  Install service: --install-service")
        

if __name__ == "__main__":
    main()

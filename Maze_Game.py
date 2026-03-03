
import numpy as np
import pygame
import sys
import random
from enum import Enum
import time

# Initialize Pygame - this is required for creating the game window
pygame.init()

class CellType(Enum):
    """
    Maze cell type enumeration.
    Each cell in the maze can be one of these types.
    """
    EMPTY = 0      # Empty space - agent can move here
    WALL = 1       # Wall - agent cannot move here
    AGENT = 2      # Current position of the agent
    TARGET = 3     # Goal position (cheese)
    PATH = 4       # Cells that agent has visited

class QLearningMaze:
    """
    Main Q-learning Maze Game Class.
    
    This class handles:
    1. Maze generation and game state
    2. Q-learning algorithm implementation
    3. Action selection and reward calculation
    4. Learning progress tracking
    """
    
    def __init__(self, width=4, height=4, maze_density=0.1):
        """
        Initialize the maze and Q-learning parameters.
        
        Args:
            width: Maze width in cells (smaller = faster learning)
            height: Maze height in cells
            maze_density: Percentage of walls (0.1 = 10% walls)
        """
        # Maze dimensions - smaller maze = faster learning
        self.width = width
        self.height = height
        self.maze_density = maze_density
        
        # ====================================
        # GAME STATE VARIABLES
        # ====================================
        self.maze = None           # 2D array representing the maze
        self.agent_pos = None      # Current (x, y) position of agent
        self.target_pos = None     # Position of the target (cheese)
        self.done = False          # Whether current episode is finished
        self.steps = 0             # Steps taken in current episode
        self.max_steps = width * height * 2  # Max steps before episode ends
        
        # ====================================
        # Q-LEARNING PARAMETERS
        # ====================================
        self.q_table = {}           # Q-table: stores action values for each state
        self.learning_rate = 0.15   # Alpha - how much to update Q-values (0-1)
                                   # Higher = faster learning, but may be unstable
        
        self.discount_factor = 0.95 # Gamma - importance of future rewards (0-1)
                                   # Higher = agent cares more about long-term rewards
        
        self.epsilon = 0.7          # Exploration rate (0-1)
                                   # Higher = more random exploration
        
        self.epsilon_decay = 0.995   # How much epsilon decreases each episode
                                   # Controls exploration vs exploitation balance
        
        self.min_epsilon = 0.05      # Minimum exploration rate
                                   # Ensures agent never stops exploring completely
        
        # ====================================
        # ACTION SPACE
        # ====================================
        # Possible actions the agent can take
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # How each action changes the agent's position
        self.action_to_delta = {
            'UP': (0, -1),     # x unchanged, y decreases
            'DOWN': (0, 1),     # x unchanged, y increases
            'LEFT': (-1, 0),    # x decreases, y unchanged
            'RIGHT': (1, 0)     # x increases, y unchanged
        }
        
        # ====================================
        # REWARD FUNCTION
        # ====================================
        # Rewards guide the agent's behavior
        # Positive rewards = good behavior
        # Negative rewards = bad behavior
        self.rewards = {
            'target': 150,    # Big reward for reaching the goal
            'wall': -15,      # Punishment for hitting walls
            'step': -1,       # Small punishment for each step (encourages efficiency)
            'repeat': -3      # Punishment for revisiting cells (prevents loops)
        }
        
        # ====================================
        # STATISTICS TRACKING
        # ====================================
        self.episode_rewards = []    # Total reward for each episode
        self.episode_lengths = []    # Number of steps in each episode
        self.success_history = []    # Whether each episode was successful (1=success, 0=fail)
        
        # For displaying current action in UI
        self.last_action = None
        self.last_reward = 0
        
        # Initialize the maze
        self.reset()
        
    def generate_maze(self):
        """
        Generate a random maze with walls.
        
        Returns:
            2D numpy array where 0=empty, 1=wall
        """
        # Start with all empty cells
        maze = np.zeros((self.height, self.width), dtype=int)
        
        # Calculate how many walls to place
        num_walls = int(self.width * self.height * self.maze_density)
        
        # Randomly place walls
        if num_walls > 0:
            # Randomly select positions for walls
            wall_positions = random.sample(
                range(self.width * self.height), 
                num_walls
            )
            
            for pos in wall_positions:
                x = pos % self.width
                y = pos // self.width
                maze[y, x] = CellType.WALL.value
        
        return maze
    
    def reset(self):
        """
        Reset the game for a new episode.
        
        This is called at the start of each training episode.
        
        Returns:
            The initial state
        """
        # Generate a new random maze
        self.maze = self.generate_maze()
        
        # Place agent and target in valid positions
        # Create list of all possible positions
        positions = list(range(self.width * self.height))
        random.shuffle(positions)  # Randomize order
        
        agent_placed = False
        target_placed = False
        
        # Find valid positions for agent and target
        for pos in positions:
            x = pos % self.width
            y = pos // self.width
            
            # Check if this cell is not a wall
            if self.maze[y, x] != CellType.WALL.value:
                if not agent_placed:
                    # Place the agent here
                    self.agent_pos = (x, y)
                    agent_placed = True
                elif not target_placed and (x, y) != self.agent_pos:
                    # Place the target here (different from agent position)
                    self.target_pos = (x, y)
                    target_placed = True
                    break
        
        # Reset episode variables
        self.done = False
        self.steps = 0
        self.visited_positions = [self.agent_pos]  # Track visited cells
        self.last_action = None
        self.last_reward = 0
        
        return self.get_state()
    
    def get_state(self):
        """
        Get the current state representation.
        
        The state includes:
        - Current position
        - Target direction
        - Distance to target
        
        Returns:
            String representing the current state
        """
        x, y = self.agent_pos
        
        # Get target direction (dx, dy)
        target_dx = self.target_pos[0] - x
        target_dy = self.target_pos[1] - y
        
        # Calculate Manhattan distance to target
        distance = abs(target_dx) + abs(target_dy)
        
        # Create state string
        # Format: "pos=x,y_target=dx,dy_dist=distance"
        state = f"pos={x},{y}_target={target_dx},{target_dy}_dist={distance}"
        
        return state
    
    def get_action(self, state, training=True):
        """
        Select an action using epsilon-greedy strategy.
        
        Epsilon-greedy balances exploration and exploitation:
        - With probability epsilon: explore (random action)
        - With probability 1-epsilon: exploit (best known action)
        
        Args:
            state: Current state
            training: Whether we're in training mode
        
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # EXPLORE: Take random action to discover new possibilities
            action = random.choice(self.actions)
            self.last_action = f"Explore: {action}"
            return action
        else:
            # EXPLOIT: Take best known action from Q-table
            if state not in self.q_table:
                # Initialize Q-values for new state
                self.q_table[state] = {action: 0 for action in self.actions}
            
            # Find action(s) with highest Q-value
            max_q = max(self.q_table[state].values())
            best_actions = [a for a, q in self.q_table[state].items() if q == max_q]
            
            # Randomly choose among best actions (in case of ties)
            action = random.choice(best_actions)
            self.last_action = f"Exploit: {action}"
            return action
    
    def take_action(self, action):
        """
        Execute an action and get the result.
        
        Args:
            action: Action to take
        
        Returns:
            next_state: New state after action
            reward: Reward received
            done: Whether episode is finished
        """
        x, y = self.agent_pos
        dx, dy = self.action_to_delta[action]
        new_x, new_y = x + dx, y + dy
        
        # ====================================
        # CHECK 1: Out of bounds?
        # ====================================
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            self.last_reward = self.rewards['wall']
            return self.get_state(), self.rewards['wall'], self.done
        
        # ====================================
        # CHECK 2: Hit a wall?
        # ====================================
        if self.maze[new_y, new_x] == CellType.WALL.value:
            self.last_reward = self.rewards['wall']
            return self.get_state(), self.rewards['wall'], self.done
        
        # ====================================
        # MOVE THE AGENT
        # ====================================
        self.agent_pos = (new_x, new_y)
        self.steps += 1
        
        # Start with step penalty
        reward = self.rewards['step']
        
        # ====================================
        # CHECK 3: Reached the target?
        # ====================================
        if self.agent_pos == self.target_pos:
            reward = self.rewards['target']  # Big reward for success
            self.done = True
            self.success_history.append(1)   # Record success
        else:
            # ====================================
            # SHAPING REWARD: Getting closer to target
            # ====================================
            # Calculate distances before and after move
            old_distance = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
            new_distance = abs(new_x - self.target_pos[0]) + abs(new_y - self.target_pos[1])
            
            # Reward for getting closer, punish for moving away
            if new_distance < old_distance:
                reward += 3   # Positive reward for moving toward target
            elif new_distance > old_distance:
                reward -= 2   # Negative reward for moving away
        
        # ====================================
        # CHECK 4: Revisiting a cell?
        # ====================================
        if self.agent_pos in self.visited_positions:
            reward += self.rewards['repeat']  # Punish loops
        else:
            self.visited_positions.append(self.agent_pos)  # Mark as visited
        
        # ====================================
        # CHECK 5: Too many steps?
        # ====================================
        if self.steps >= self.max_steps:
            self.done = True
            if not self.agent_pos == self.target_pos:
                self.success_history.append(0)  # Record failure
        
        self.last_reward = reward
        return self.get_state(), reward, self.done
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-values using the Q-learning formula.
        
        Q-learning formula:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        Where:
        - Q(s,a): Current Q-value for state s and action a
        - α: Learning rate
        - r: Reward received
        - γ: Discount factor
        - max(Q(s',a')): Maximum Q-value for next state
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Initialize Q-values if states are new
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update formula
        # new_q = current_q + learning_rate * (reward + discount * max_next_q - current_q)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def train_step(self):
        """
        Perform one training step.
        
        Returns:
            True if episode is done, False otherwise
        """
        if self.done:
            return True
        
        # Get current state
        state = self.get_state()
        
        # Choose action (with exploration)
        action = self.get_action(state, training=True)
        
        # Take action and observe result
        next_state, reward, done = self.take_action(action)
        
        # Update Q-table based on experience
        self.update_q_table(state, action, reward, next_state)
        
        return done
    
    def test_step(self):
        """
        Perform one testing step (no learning).
        
        Returns:
            True if episode is done, False otherwise
        """
        if self.done:
            return True
        
        # Get current state
        state = self.get_state()
        
        # Choose action (no exploration - use best action)
        action = self.get_action(state, training=False)
        
        # Take action
        next_state, reward, done = self.take_action(action)
        
        return done
    
    def get_success_rate(self, window=15):
        """
        Calculate success rate over recent episodes.
        
        Args:
            window: Number of recent episodes to consider
        
        Returns:
            Success rate as percentage (0-100)
        """
        if len(self.success_history) < 5:
            return 0.0
        recent = self.success_history[-min(window, len(self.success_history)):]
        return sum(recent) / len(recent) * 100


class MazeGUI:
    """
    Graphical User Interface for the Maze Game.
    
    This class handles:
    1. Drawing the maze and game elements
    2. Button interactions
    3. Displaying statistics
    4. Controlling training speed
    """
    
    def __init__(self, maze_game, cell_size=90):
        """
        Initialize the GUI.
        
        Args:
            maze_game: QLearningMaze instance
            cell_size: Size of each cell in pixels
        """
        self.game = maze_game
        self.cell_size = cell_size
        
        # ====================================
        # WINDOW SETUP
        # ====================================
        # Calculate window dimensions
        self.window_width = max(1100, maze_game.width * cell_size + 300)
        self.window_height = maze_game.height * cell_size + 280
        
        # Create pygame window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Q-learning Maze Game - Explained Version")
        
        # ====================================
        # COLOR DEFINITIONS
        # ====================================
        self.colors = {
            CellType.EMPTY: (255, 255, 255),  # White
            CellType.WALL: (0, 0, 0),          # Black
            CellType.AGENT: (0, 0, 255),        # Blue
            CellType.TARGET: (255, 0, 0),       # Red
            CellType.PATH: (200, 200, 255),     # Light blue
            'grid': (200, 200, 200),             # Gray
            'text': (0, 0, 0),                    # Black
            'button': (100, 200, 100),            # Green
            'button_hover': (150, 255, 150),      # Light green
            'info_bg': (240, 240, 240)            # Light gray
        }
        
        # ====================================
        # FONTS
        # ====================================
        self.font = pygame.font.Font(None, 26)        # Regular text
        self.title_font = pygame.font.Font(None, 38)  # Title
        self.small_font = pygame.font.Font(None, 22)  # Small text
        
        # ====================================
        # BUTTON SETUP
        # ====================================
        button_width = 110
        button_height = 45
        button_y = self.window_height - 90
        start_x = 80
        spacing = 140
        
        self.buttons = {
            'train': pygame.Rect(start_x, button_y, button_width, button_height),
            'stop': pygame.Rect(start_x + spacing, button_y, button_width, button_height),
            'test': pygame.Rect(start_x + spacing * 2, button_y, button_width, button_height),
            'reset': pygame.Rect(start_x + spacing * 3, button_y, button_width, button_height),
            'faster': pygame.Rect(start_x + spacing * 4, button_y, button_width, button_height),
            'slower': pygame.Rect(start_x + spacing * 5, button_y, button_width, button_height)
        }
        
        # ====================================
        # TRAINING STATE
        # ====================================
        self.training = False
        self.episode_count = 0
        
        # ====================================
        # SPEED SETTINGS
        # ====================================
        self.step_delay = 120      # Milliseconds per step (higher = slower)
        self.steps_per_frame = 2    # Steps to process per frame
        self.frame_rate = 15        # Frames per second
        
        # ====================================
        # STATISTICS TRACKING
        # ====================================
        self.stats = {
            'episodes': 0,           # Total episodes completed
            'success_rate': 0,        # Success percentage
            'avg_reward': 0,          # Average reward per episode
            'epsilon': self.game.epsilon,  # Current exploration rate
            'q_table_size': 0,        # Number of states in Q-table
            'current_step': 0,         # Steps in current episode
            'steps_per_second': 0      # Training speed
        }
        
        # For performance tracking
        self.last_print_time = time.time()
        self.last_success_rate = 0
        self.step_counter = 0
        self.last_step_count = 0
    
    def draw_maze(self):
        """
        Draw the maze on the screen.
        
        This function:
        1. Centers the maze in the window
        2. Draws each cell with appropriate color
        3. Adds grid lines for clarity
        """
        # Calculate maze dimensions and center position
        maze_width = self.game.width * self.cell_size
        offset_x = (self.window_width - maze_width) // 2
        
        # Draw each cell
        for y in range(self.game.height):
            for x in range(self.game.width):
                # Calculate cell rectangle
                rect = pygame.Rect(
                    offset_x + x * self.cell_size, 
                    y * self.cell_size,
                    self.cell_size, 
                    self.cell_size
                )
                
                # Determine cell color based on type
                cell_value = self.game.maze[y, x]
                
                if (x, y) == self.game.agent_pos:
                    color = self.colors[CellType.AGENT]  # Blue = agent
                elif (x, y) == self.game.target_pos:
                    color = self.colors[CellType.TARGET]  # Red = target
                elif hasattr(self.game, 'visited_positions') and (x, y) in self.game.visited_positions:
                    color = self.colors[CellType.PATH]  # Light blue = visited
                elif cell_value == CellType.WALL.value:
                    color = self.colors[CellType.WALL]  # Black = wall
                else:
                    color = self.colors[CellType.EMPTY]  # White = empty
                
                # Draw cell and grid line
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 2)
    
    def draw_info(self):
        """
        Draw information panel with statistics and controls.
        
        This displays:
        1. Current action and reward
        2. Learning statistics
        3. Progress bar
        4. Button controls
        """
        info_y = self.game.height * self.cell_size + 20
        
        # ====================================
        # TITLE
        # ====================================
        title = self.title_font.render("Q-learning Maze Game", True, self.colors['text'])
        title_rect = title.get_rect(center=(self.window_width // 2, info_y))
        self.screen.blit(title, title_rect)
        
        # ====================================
        # CURRENT ACTION AND REWARD
        # ====================================
        action_text = f"Action: {self.game.last_action if self.game.last_action else 'Waiting'}"
        reward_text = f"Reward: {self.game.last_reward:.1f}"
        
        action_surface = self.font.render(action_text, True, (0, 0, 150))
        reward_surface = self.font.render(reward_text, True, (150, 0, 0) if self.game.last_reward < 0 else (0, 150, 0))
        
        self.screen.blit(action_surface, (50, info_y + 45))
        self.screen.blit(reward_surface, (450, info_y + 45))
        
        # ====================================
        # UPDATE STATISTICS
        # ====================================
        stats_y = info_y + 90
        
        # Update Q-table size and current step
        self.stats['q_table_size'] = len(self.game.q_table)
        self.stats['epsilon'] = self.game.epsilon
        self.stats['current_step'] = self.game.steps
        
        # Calculate success rate
        self.stats['success_rate'] = self.game.get_success_rate(15)
        self.stats['episodes'] = len(self.game.success_history)
        
        # Calculate average reward
        if len(self.game.episode_rewards) > 0:
            recent_rewards = self.game.episode_rewards[-15:]
            self.stats['avg_reward'] = np.mean(recent_rewards) if recent_rewards else 0
        
        # Calculate steps per second
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            self.stats['steps_per_second'] = self.step_counter - self.last_step_count
            self.last_step_count = self.step_counter
            self.last_print_time = current_time
        
        # Print progress when success rate changes significantly
        if abs(self.stats['success_rate'] - self.last_success_rate) > 8:
            self.last_success_rate = self.stats['success_rate']
            print(f"Episode {self.stats['episodes']}: Success Rate = {self.stats['success_rate']:.1f}%")
        
        # ====================================
        # STATISTICS DISPLAY - THREE COLUMNS
        # ====================================
        col1_x = 80
        col2_x = 380
        col3_x = 680
        
        # Column 1: Episode statistics
        info_texts_col1 = [
            f"Episodes: {self.stats['episodes']}",
            f"Success Rate: {self.stats['success_rate']:.1f}%",
            f"Epsilon: {self.stats['epsilon']:.3f}"
        ]
        
        # Column 2: Learning statistics
        info_texts_col2 = [
            f"Q-table: {self.stats['q_table_size']}",
            f"Steps: {self.stats['current_step']}",
            f"Avg Reward: {self.stats['avg_reward']:.1f}"
        ]
        
        # Column 3: Speed settings
        info_texts_col3 = [
            f"Delay: {self.step_delay}ms",
            f"Steps/Frame: {self.steps_per_frame}",
            f"Steps/sec: {self.stats['steps_per_second']}"
        ]
        
        # Draw column 1
        for i, text in enumerate(info_texts_col1):
            info = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(info, (col1_x, stats_y + i * 28))
        
        # Draw column 2
        for i, text in enumerate(info_texts_col2):
            info = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(info, (col2_x, stats_y + i * 28))
        
        # Draw column 3
        for i, text in enumerate(info_texts_col3):
            info = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(info, (col3_x, stats_y + i * 28))
        
        # ====================================
        # MODE INDICATOR
        # ====================================
        mode_y = stats_y + 100
        if self.training:
            mode_text = "▶ TRAINING MODE - Learning in Progress"
            mode_color = (0, 150, 0)
        else:
            mode_text = "⏸ PAUSED - Press TRAIN to continue"
            mode_color = (150, 0, 0)
        
        mode_surface = self.font.render(mode_text, True, mode_color)
        mode_rect = mode_surface.get_rect(center=(self.window_width // 2, mode_y))
        self.screen.blit(mode_surface, mode_rect)
        
        # ====================================
        # PROGRESS BAR
        # ====================================
        bar_y = mode_y + 35
        bar_x = 200
        bar_width = 700
        bar_height = 25
        
        # Background bar
        pygame.draw.rect(self.screen, (220, 220, 220), (bar_x, bar_y, bar_width, bar_height))
        
        # Colored progress based on success rate
        success_width = int(bar_width * self.stats['success_rate'] / 100)
        if self.stats['success_rate'] >= 80:
            bar_color = (0, 200, 0)      # Green = expert
        elif self.stats['success_rate'] >= 50:
            bar_color = (200, 200, 0)    # Yellow = intermediate
        elif self.stats['success_rate'] >= 20:
            bar_color = (200, 100, 0)    # Orange = beginner
        else:
            bar_color = (200, 0, 0)       # Red = just starting
        
        # Draw progress
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, success_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Show percentage on bar if enough space
        if success_width > 50:
            bar_text = self.small_font.render(f"{self.stats['success_rate']:.1f}%", True, (255, 255, 255))
            text_rect = bar_text.get_rect(center=(bar_x + success_width//2, bar_y + bar_height//2))
            self.screen.blit(bar_text, text_rect)
        
        # Milestone markers
        for percent in [25, 50, 75]:
            marker_x = bar_x + int(bar_width * percent / 100)
            pygame.draw.line(self.screen, (100, 100, 100), (marker_x, bar_y - 5), (marker_x, bar_y + bar_height + 5), 2)
            marker_text = self.small_font.render(f"{percent}%", True, (100, 100, 100))
            self.screen.blit(marker_text, (marker_x - 15, bar_y - 22))
        
        # ====================================
        # DRAW BUTTONS
        # ====================================
        mouse_pos = pygame.mouse.get_pos()
        
        for name, rect in self.buttons.items():
            # Highlight active buttons
            if name == 'train' and self.training:
                color = (0, 200, 0)  # Green when training
            elif name == 'stop' and not self.training:
                color = (200, 0, 0)  # Red when paused
            else:
                color = self.colors['button_hover'] if rect.collidepoint(mouse_pos) else self.colors['button']
            
            # Draw button
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.colors['text'], rect, 2)
            
            # Button text
            display_names = {
                'train': 'TRAIN',
                'stop': 'STOP',
                'test': 'TEST',
                'reset': 'RESET',
                'faster': 'FASTER',
                'slower': 'SLOWER'
            }
            
            btn_text = self.font.render(display_names[name], True, self.colors['text'])
            text_rect = btn_text.get_rect(center=rect.center)
            self.screen.blit(btn_text, text_rect)
    
    def handle_click(self, pos):
        """
        Handle mouse clicks on buttons.
        
        Args:
            pos: Mouse click position (x, y)
        
        Returns:
            True if a button was clicked, False otherwise
        """
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if name == 'train':
                    # Start training
                    self.training = True
                    self.game.last_action = "Training started"
                elif name == 'stop':
                    # Pause training
                    self.training = False
                    self.game.last_action = "Training paused"
                elif name == 'test':
                    # Test the trained agent
                    self.test_agent()
                elif name == 'reset':
                    # Reset the game
                    self.reset_game()
                elif name == 'faster':
                    # Increase speed (decrease delay, increase steps)
                    self.step_delay = max(50, self.step_delay - 20)
                    self.steps_per_frame = min(4, self.steps_per_frame + 1)
                    self.game.last_action = f"Speed: {self.step_delay}ms, {self.steps_per_frame} steps/frame"
                elif name == 'slower':
                    # Decrease speed (increase delay, decrease steps)
                    self.step_delay = min(300, self.step_delay + 20)
                    self.steps_per_frame = max(1, self.steps_per_frame - 1)
                    self.game.last_action = f"Speed: {self.step_delay}ms, {self.steps_per_frame} steps/frame"
                return True
        return False
    
    def test_agent(self):
        """
        Test the trained agent without learning.
        
        This shows how well the agent has learned by
        letting it run using its best actions (no exploration).
        """
        self.training = False
        if self.game.done:
            self.game.reset()
        
        self.game.last_action = "Testing agent..."
        
        # Test at comfortable speed
        test_delay = 150
        while not self.game.done:
            self.game.test_step()
            self.draw_maze()
            self.draw_info()
            pygame.display.flip()
            pygame.time.wait(test_delay)
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.game.reset()
        self.training = False
        self.game.last_action = "Game reset"
        self.game.last_reward = 0
        self.last_success_rate = 0
        self.step_counter = 0
    
    def update(self):
        """Update game state during training."""
        if not self.training:
            return
        
        # Train multiple steps per frame
        for _ in range(self.steps_per_frame):
            if self.game.done:
                self.game.reset()
            else:
                self.game.train_step()
                self.step_counter += 1
        
        # Delay to control speed
        if self.step_delay > 0:
            pygame.time.wait(self.step_delay)
        
        # Decay epsilon (exploration rate)
        if self.game.epsilon > self.game.min_epsilon:
            self.game.epsilon *= self.game.epsilon_decay
    
    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        # Print welcome message
        print("\n" + "=" * 60)
        print("Q-LEARNING MAZE GAME - EXPLAINED VERSION")
        print("=" * 60)
        print("\n🎯 GAME OBJECTIVE:")
        print("   The blue agent must learn to find the red target (cheese)")
        print("   while avoiding black walls.")
        print("\n🧠 HOW IT LEARNS:")
        print("   • Exploration: Tries random actions (high epsilon)")
        print("   • Exploitation: Uses learned Q-values (low epsilon)")
        print("   • Reward: Gets +150 for target, -15 for walls")
        print("\n⚡ CURRENT SPEED SETTINGS:")
        print(f"   • Delay: {self.step_delay}ms per step")
        print(f"   • Steps per frame: {self.steps_per_frame}")
        print(f"   • Target speed: 15-20 steps/second")
        print("\n📈 EXPECTED PROGRESS (1 minute):")
        print("   0-15s:  Exploring (0-30% success)")
        print("   15-30s: Learning (30-60% success)")
        print("   30-45s: Improving (60-80% success)")
        print("   45-60s: Mastering (80%+ success)")
        print("\n🎮 CONTROLS:")
        print("   TRAIN   - Start training")
        print("   STOP    - Pause training")
        print("   TEST    - Watch trained agent")
        print("   RESET   - Start over")
        print("   FASTER  - Increase speed")
        print("   SLOWER  - Decrease speed")
        print("   SPACEBAR- Manual step when paused")
        print("   ESC     - Exit")
        print("=" * 60 + "\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Manual step when paused
                        if not self.training and not self.game.done:
                            self.game.train_step()
                            self.step_counter += 1
            
            # Update game state
            self.update()
            
            # Draw everything
            self.screen.fill((240, 240, 240))
            self.draw_maze()
            self.draw_info()
            
            # Update display
            pygame.display.flip()
            clock.tick(self.frame_rate)
        
        pygame.quit()
        sys.exit()


def main():
    """Main function - entry point of the program."""
    
    # Create game instance
    # 4x4 maze with 10% walls for fast learning
    maze_game = QLearningMaze(width=4, height=4, maze_density=0.1)
    
    # Create GUI with large cells for better visibility
    gui = MazeGUI(maze_game, cell_size=100)
    
    # Run the game
    gui.run()


if __name__ == "__main__":
    main()
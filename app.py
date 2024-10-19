#import pygame
import numpy as np
import time

# 棋子代碼：帥=1，士=2，象=3，馬=4，車=5，炮=6，兵=7
# 紅方正數，黑方負數

def initialize_board():
    board = np.zeros((10, 9), dtype=int)

    # 初始化紅方棋子
    board[0] = [5, 4, 3, 2, 1, 2, 3, 4, 5]  # 車馬象士帥士象馬車
    board[2][1] = board[2][7] = 6  # 紅方炮
    board[3][0] = board[3][2] = board[3][4] = board[3][6] = board[3][8] = 7  # 紅方兵

    # 初始化黑方棋子
    board[9] = [-5, -4, -3, -2, -1, -2, -3, -4, -5]  # 車馬象士將士象馬車
    board[7][1] = board[7][7] = -6  # 黑方炮
    board[6][0] = board[6][2] = board[6][4] = board[6][6] = board[6][8] = -7  # 黑方卒

    return board

def display_board(board):
    for row in board:
        print(' '.join(f'{x:2}' for x in row))

# 定義棋盤參數
GRID_SIZE = 60  # 棋格大小
BOARD_WIDTH = 9  # 棋盤寬
BOARD_HEIGHT = 10  # 棋盤高

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def render_chess_board(screen, state):
    screen.fill((255, 255, 255))  # 清空畫面，設置背景顏色

    # 遍歷棋盤，繪製每個棋子
    for x in range(10):  # 中國象棋 10 行
        for y in range(9):  # 中國象棋 9 列
            piece = state[x][y]
            if piece != 0:
                draw_piece(screen, x, y, piece)

# 定義棋子的類型映射
piece_names = {
    1: "帥", 2: "仕", 3: "相", 4: "傌", 5: "俥", 6: "炮", 7: "兵",  # 紅方棋子
    -1: "將", -2: "士", -3: "象", -4: "馬", -5: "車", -6: "包", -7: "卒"  # 藍方棋子
}

def draw_piece(screen, x, y, piece):
    # 根據 `piece` 繪製不同的棋子
    color = (255, 0, 0) if piece > 0 else (0, 0, 255)  # 紅色: 紅方, 藍色: 藍方
    piece_radius = 20  # 棋子的半徑大小

    # 畫出圓形棋子
    pygame.draw.circle(screen, color, (y * 60 + 30, x * 60 + 30), piece_radius)

    # 獲取棋子的名稱
    piece_name = piece_names.get(piece, "")

    # 設置字體和大小
    font = pygame.font.Font("Fonts/msjh.ttc", 20)
    
    # 渲染棋子的名稱
    text = font.render(piece_name, True, (0, 0, 0))  # 黑色文字
    text_rect = text.get_rect(center=(y * 60 + 30, x * 60 + 30))

    # 將文字渲染到畫布上
    screen.blit(text, text_rect)



# 設定每個棋子的 reward 值
def calculate_reward(piece):
    if abs(piece) == 7:  # 兵卒
        return 0.5
    elif abs(piece) == 4 or abs(piece) == 3:  # 馬象
        return 2.0
    elif abs(piece) == 1:  # 帥將
        return 10.0
    return 0  # 沒有吃子

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定義CNN架構，用於處理棋盤狀態並預測Q值
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 卷積層，輸入棋盤狀態 (10x9)，2層CNN進行提取特徵
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 9, 512)  # 全連接層
        self.fc2 = nn.Linear(512, 90 * 90)  # 棋盤上90個位置的Q值 (90*90是所有可能的棋子移動組合)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平張量
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # 輸出所有動作的Q值

# 建立 DQN 模型
dqn_red = DQN()
dqn_blue = DQN()

# 優化器
optimizer_red = optim.Adam(dqn_red.parameters(), lr=0.001)
optimizer_blue = optim.Adam(dqn_blue.parameters(), lr=0.001)

# 損失函數
criterion = nn.MSELoss()

# 超參數
GAMMA = 0.99  # 折扣因子
EPSILON = 1.0  # 初始探索率
EPSILON_MIN = 0.1  # 最小探索率
EPSILON_DECAY = 0.995  # 探索率衰減
BATCH_SIZE = 32  # 每次更新的樣本數
REPLAY_MEMORY_SIZE = 10000  # 經驗回放緩衝區大小
TARGET_UPDATE_FREQUENCY = 100  # 更新目標網路的頻率

# 經驗回放記憶體
memory = deque(maxlen=REPLAY_MEMORY_SIZE)

# 選擇行動：使用 ε-greedy 策略
def select_action(state, epsilon, model):
    if np.random.rand() <= epsilon:
        # 探索：隨機選擇一個行動
        return random.randint(0, 90 * 90 - 1)
    else:
        # 利用：選擇Q值最大的行動
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # 增加維度適應CNN
            q_values = model(state)
            return q_values.argmax().item()

# 存儲經驗
def store_experience(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# DQN訓練函數
def train_dqn(model, target_model, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    
    # 從記憶體中隨機抽取樣本
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # 預測Q值
    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)

    # 計算期望Q值
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    # 更新神經網路
    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 更新目標網路
def update_target_model(model, target_model):
    target_model.load_state_dict(model.state_dict())
# 假設棋盤是 10x9 的 numpy 陣列，棋子用數字表示 (正數表示紅方，負數表示藍方)
# 例如：1 代表紅方將帥，-1 代表藍方將帥，其他棋子類似
# 行動 (action) 是一個整數，對應於 90 個可能位置中某個棋子的移動

def make_move(state, action, player_color):
    # 將 action 轉換為具體的起始位置和目標位置
    start_pos = action // 90  # 90x90 是所有可能的動作組合
    target_pos = action % 90

    start_x, start_y = divmod(start_pos, 9)
    target_x, target_y = divmod(target_pos, 9)

    piece = state[start_x][start_y]
    target_piece = state[target_x][target_y]

    # 確認是否選擇了合法的棋子
    if (piece == 0 or 
       (player_color == "red" and piece < 0) or 
       (player_color == "blue" and piece > 0)):
        return state, -5, False  # 選擇了對方的棋子或空格

    # 確認是否移動到自己的棋子上
    if (player_color == "red" and target_piece > 0) or (player_color == "blue" and target_piece < 0):
        return state, -5, False  # 不能吃自己的棋子

    # 檢查棋子的具體移動規則
    if not is_valid_move(piece, start_x, start_y, target_x, target_y, state):
        return state, -5, False  # 非法移動

    # 移動棋子
    state[target_x][target_y] = piece
    state[start_x][start_y] = 0

    # 計算 reward（吃子得分）
    reward = calculate_reward(target_piece)

    # 判斷是否遊戲結束
    done = False
    if abs(target_piece) == 1:  # 將帥被吃
        done = True
        reward += 10  # 加上額外的勝利分數

    return state, reward, done

def is_valid_move(piece, start_x, start_y, target_x, target_y, state):
    # 獲取目標位置的棋子
    target_piece = state[target_x][target_y]

    # 檢查目標位置是否越界
    if target_x < 0 or target_x >= 10 or target_y < 0 or target_y >= 9:
        return False
    
    # 取得棋子類型，正數代表紅方，負數代表藍方
    abs_piece = abs(piece)

    # 帥（將）移動規則
    if abs_piece == 1:
        if target_x < 7 or target_x > 9 or target_y < 3 or target_y > 5:  # 限定在九宮內
            return False
        if abs(start_x - target_x) + abs(start_y - target_y) != 1:  # 只能橫或直走一步
            return False

    # 士（仕）的移動規則
    elif abs_piece == 2:
        if target_x < 7 or target_x > 9 or target_y < 3 or target_y > 5:  # 只能在九宮內走對角
            return False
        if abs(start_x - target_x) != 1 or abs(start_y - target_y) != 1:  # 只能斜走一步
            return False

    # 象（相）的移動規則
    elif abs_piece == 3:
        if target_x > 4:  # 相不能過河
            return False
        if abs(start_x - target_x) != 2 or abs(start_y - target_y) != 2:  # 走田字
            return False
        if state[(start_x + target_x) // 2][(start_y + target_y) // 2] != 0:  # 象眼被堵住
            return False

    # 馬（傌）的移動規則
    elif abs_piece == 4:
        dx = abs(start_x - target_x)
        dy = abs(start_y - target_y)
        if dx == 2 and dy == 1:
            if state[(start_x + target_x) // 2][start_y] != 0:  # 拐馬腳被堵住
                return False
        elif dx == 1 and dy == 2:
            if state[start_x][(start_y + target_y) // 2] != 0:  # 拐馬腳被堵住
                return False
        else:
            return False

    # 車（俥）的移動規則
    elif abs_piece == 5:
        if start_x != target_x and start_y != target_y:
            return False  # 只能直走
        if start_x == target_x:  # 橫向移動
            for y in range(min(start_y, target_y) + 1, max(start_y, target_y)):
                if state[start_x][y] != 0:
                    return False  # 中途有棋子
        else:  # 縱向移動
            for x in range(min(start_x, target_x) + 1, max(start_x, target_x)):
                if state[x][start_y] != 0:
                    return False  # 中途有棋子

    # 炮（炮）的移動規則
    elif abs_piece == 6:
        if start_x != target_x and start_y != target_y:
            return False  # 只能直走
        count = 0
        if start_x == target_x:  # 橫向移動
            for y in range(min(start_y, target_y) + 1, max(start_y, target_y)):
                if state[start_x][y] != 0:
                    count += 1
        else:  # 縱向移動
            for x in range(min(start_x, target_x) + 1, max(start_x, target_x)):
                if state[x][start_y] != 0:
                    count += 1
        if count == 0 and target_piece == 0:
            return True  # 炮空行無子
        elif count == 1 and target_piece != 0:
            return True  # 炮吃子需隔一個棋子
        return False

    # 兵（卒）的移動規則
    elif abs_piece == 7:
        if piece > 0 and start_x <= 4:  # 紅兵未過河
            if target_x != start_x + 1 or target_y != start_y:
                return False
        elif piece < 0 and start_x >= 5:  # 黑卒未過河
            if target_x != start_x - 1 or target_y != start_y:
                return False
        else:  # 過河後可以左右移動
            if abs(start_x - target_x) + abs(start_y - target_y) != 1:
                return False

    return True

# 模擬一場比賽
def play_game():
    state = initialize_board()  # 初始棋盤狀態

    # Pygame 初始化
    #pygame.init() 
    #screen = pygame.display.set_mode((GRID_SIZE * BOARD_WIDTH, GRID_SIZE * BOARD_HEIGHT))
    #pygame.display.set_caption("RL 中國象棋訓練過程")

    done = False
    total_reward_red = 0
    total_reward_blue = 0

    for epoch in range(10000):  # 訓練多個回合
        epsilon = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # Red Agent 的回合
        action_red = select_action(state, epsilon, dqn_red)
        next_state_red, reward_red, done_red = make_move(state, action_red, "red")
        total_reward_red += reward_red

        # 日誌輸出紅方的決策
        #print(f"紅方動作: {action_red}, 獲得獎勵: {reward_red}, \n新狀態:\n {next_state_red}\nred end")
        print(f"紅方動作: {action_red}, 獲得獎勵: {reward_red}")
        store_experience(memory, state, action_red, reward_red, next_state_red, done_red)
        state = next_state_red

        # 更新紅方模型
        train_dqn(dqn_red, dqn_red, memory, optimizer_red)

        # 更新 pygame 畫面
        #render_chess_board(screen, state)  # 繪製當前棋盤狀態
        #pygame.display.update()  # 刷新顯示

        if done_red:
            break

        # Blue Agent 的回合
        action_blue = select_action(state, epsilon, dqn_blue)
        next_state_blue, reward_blue, done_blue = make_move(state, action_blue, "blue")
        total_reward_blue += reward_blue

        # 日誌輸出藍方的決策
        #print(f"藍方動作: {action_blue}, 獲得獎勵: {reward_blue}, \n新狀態:\n {next_state_blue}\nblue end")
        print(f"藍方動作: {action_blue}, 獲得獎勵: {reward_blue}")

        store_experience(memory, state, action_blue, reward_blue, next_state_blue, done_blue)
        state = next_state_blue

        # 更新藍方模型
        train_dqn(dqn_blue, dqn_blue, memory, optimizer_blue)

        # 更新 pygame 畫面
        #render_chess_board(screen, state)
        #pygame.display.update()

        if done_blue:
            break

        # 每個 epoch 結束後，打印得分
        print(f'Epoch {epoch+1}, 紅方總分: {total_reward_red}, 藍方總分: {total_reward_blue}')

        # 更新目標網路
        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            update_target_model(dqn_red, dqn_red)
            update_target_model(dqn_blue, dqn_blue)

    print(f'紅方總分: {total_reward_red}, 藍方總分: {total_reward_blue}')
    #pygame.quit()


# 運行遊戲
play_game()

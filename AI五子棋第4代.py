import pygame
import sys
import math
import random
import time
import threading
from queue import Queue
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from collections import Counter, defaultdict

print('\n源码：https://github.com/my-txz/AI-\npygame提供GUI显示\n问题、算法反馈：https://github.com/my-txz/AI-/issues/\n')

# ==== 基础参数 ====
BOARD_SIZE = 15
CELL_SIZE = 32
MARGIN = 24
BTN_HEIGHT = 32
BTN_WIDTH = 120
BTN_SPACING = 12
MODES_PER_ROW = 3
TOP_PANEL_HEIGHT = 60
BTN_PANEL_HEIGHT = 100
INFO_HEIGHT = 70
SCREEN_SIZE_X = BOARD_SIZE * CELL_SIZE + MARGIN * 2
SCREEN_SIZE_Y = TOP_PANEL_HEIGHT + BOARD_SIZE * CELL_SIZE + MARGIN * 2 + BTN_PANEL_HEIGHT + INFO_HEIGHT

FPS = 40
TIME_LIMIT = 60  # seconds

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BG_COLOR = (204, 186, 136)
BTN_COLOR = (90, 160, 220)
BTN_COLOR2 = (100, 100, 100)
BTN_STOP_COLOR = (200, 80, 60)

HUMAN = 1
AI = 2

def ask_beta_mode():
    while True:
        ans = input("是否启用Beta模式(包含：CPU AND GPU Accelerate(Beta)、AI VS AI(Beta)、Professional duel(Beta)，Beta模式有未发现的BUG和算法异常)？(Yes/No)：").strip().lower()
        if ans in ['yes', 'y']:
            return True
        elif ans in ['no', 'n']:
            return False
        else:
            print("请输入 Yes 或 No。")

BETA_MODE = ask_beta_mode()

if BETA_MODE:
    MODES = [
        ("Easy", "easy"),
        ("Normal", "normal"),
        ("Difficult", "difficult"),
        ("Hell", "hell"),
        ("Professional duel (Beta)", "pro"),
        ("AI VS AI (Beta)", "ai_vs_ai"),
    ]
else:
    MODES = [
        ("Easy", "easy"),
        ("Normal", "normal"),
        ("Difficult", "difficult"),
        ("Hell", "hell"),
    ]

def other(player):
    return HUMAN if player == AI else AI

def draw_board(screen, board):
    offset_y = TOP_PANEL_HEIGHT
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (MARGIN + CELL_SIZE * i, offset_y + MARGIN), (MARGIN + CELL_SIZE * i, offset_y + MARGIN + CELL_SIZE * (BOARD_SIZE-1)), 1)
        pygame.draw.line(screen, BLACK, (MARGIN, offset_y + MARGIN + CELL_SIZE * i), (MARGIN + CELL_SIZE * (BOARD_SIZE-1), offset_y + MARGIN + CELL_SIZE * i), 1)
    for i in [3, 7, 11]:
        for j in [3, 7, 11]:
            pygame.draw.circle(screen, BLACK, (MARGIN + CELL_SIZE * i, offset_y + MARGIN + CELL_SIZE * j), 4)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cx = MARGIN + CELL_SIZE * c
            cy = offset_y + MARGIN + CELL_SIZE * r
            if board[r][c] == HUMAN:
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL_SIZE // 2 - 2)
            elif board[r][c] == AI:
                pygame.draw.circle(screen, WHITE, (cx, cy), CELL_SIZE // 2 - 2)
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL_SIZE // 2 - 2, 1)

def draw_replay_button(screen, font, y_base):
    btn_rect = pygame.Rect(SCREEN_SIZE_X - BTN_WIDTH - BTN_SPACING, y_base, BTN_WIDTH, BTN_HEIGHT)
    pygame.draw.rect(screen, BTN_COLOR, btn_rect, border_radius=8)
    txt = font.render("Replay", True, (255,255,255))
    screen.blit(txt, (btn_rect.x+30, btn_rect.y+6))
    return btn_rect

def draw_stop_ai_duel_button(screen, font, y_base):
    btn_rect = pygame.Rect(SCREEN_SIZE_X - BTN_WIDTH - BTN_SPACING, y_base, BTN_WIDTH, BTN_HEIGHT)
    pygame.draw.rect(screen, BTN_STOP_COLOR, btn_rect, border_radius=8)
    txt = font.render("Stop AI Duel", True, (255,255,255))
    screen.blit(txt, (btn_rect.x+8, btn_rect.y+6))
    return btn_rect

def draw_mode_buttons(screen, font, current_idx, y_base):
    btns = []
    btn_row = 0
    btn_col = 0
    for idx, (label, _) in enumerate(MODES):
        btn_row = idx // MODES_PER_ROW
        btn_col = idx % MODES_PER_ROW
        bx = MARGIN + (BTN_WIDTH + BTN_SPACING) * btn_col
        by = y_base + (BTN_HEIGHT + BTN_SPACING) * btn_row
        btn_rect = pygame.Rect(bx, by, BTN_WIDTH, BTN_HEIGHT)
        c = (BTN_COLOR if idx != current_idx else (220,60,60))
        pygame.draw.rect(screen, c, btn_rect, border_radius=8)
        txt = font.render(label, True, (255,255,255))
        screen.blit(txt, (btn_rect.x + 8, btn_rect.y + 6))
        btns.append(btn_rect)
    return btns

def board_full(board):
    return all(board[r][c] != 0 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))

def check_win(board, player):
    directions = [(1,0), (0,1), (1,1), (1,-1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != player:
                continue
            for dr, dc in directions:
                cnt=1
                for k in range(1, 5):
                    nr, nc = r+dr*k, c+dc*k
                    if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==player:
                        cnt+=1
                    else:
                        break
                if cnt>=5:
                    return True
    return False

def get_moves(board):
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 0:
                found = False
                for dr in range(-2,3):
                    for dc in range(-2,3):
                        nr, nc = r+dr, c+dc
                        if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]!=0:
                            found = True
                if found or (r==BOARD_SIZE//2 and c==BOARD_SIZE//2):
                    moves.append((r,c))
    return moves if moves else [(BOARD_SIZE//2, BOARD_SIZE//2)]

def move_board(board, move, player):
    new_board = deepcopy(board)
    new_board[move[0]][move[1]] = player
    return new_board

def log(msg):
    print("[AI]", msg)

#########################################
# 改进威胁检测
def threat_analysis(board, player):
    '''
    改进威胁检测：识别活三、活四、双三、双四、冲四（如XOOOO_/_OOOOX）、眠三等
    '''
    lines = []
    for r in range(BOARD_SIZE):
        lines.append(board[r])
    for c in range(BOARD_SIZE):
        lines.append([board[r][c] for r in range(BOARD_SIZE)])
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        lines.append([board[r][r-d] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=r-d<BOARD_SIZE])
        lines.append([board[r][BOARD_SIZE-1-(r-d)] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=BOARD_SIZE-1-(r-d)<BOARD_SIZE])

    h3, h4, d3, d4 = 0, 0, 0, 0
    rush4, sleep3 = 0, 0
    threes = []
    fours = []
    rush4s = []
    sleep3s = []
    for line in lines:
        s = ''.join(str(x) for x in line)
        # 活三
        if f'0{player}{player}{player}0' in s:
            h3 += s.count(f'0{player}{player}{player}0')
            threes += [0]*s.count(f'0{player}{player}{player}0')
        # 活四
        if f'0{player}{player}{player}{player}0' in s:
            h4 += s.count(f'0{player}{player}{player}{player}0')
            fours += [0]*s.count(f'0{player}{player}{player}{player}0')
        # 冲四（_OOOOX 或 XOOOO_)
        rush4_pat = [f'0{player}{player}{player}{player}1',
                     f'1{player}{player}{player}{player}0']
        for pat in rush4_pat:
            if pat in s:
                rush4 += s.count(pat)
                rush4s += [0]*s.count(pat)
        # 眠三：如 011100、001110、011010、010110
        sleep3_pat = [
            f'0{player}{player}{player}00',
            f'00{player}{player}{player}0',
            f'0{player}{player}0{player}0',
            f'0{player}0{player}{player}0',
        ]
        for pat in sleep3_pat:
            if pat in s:
                sleep3 += s.count(pat)
                sleep3s += [0]*s.count(pat)
    # 双三/双四判定
    if h3 >= 2: d3 = 1
    if h4 >= 2: d4 = 1
    # 冲四和眠三也输出
    return h3, h4, d3, d4, rush4, sleep3

#########################################
def heuristic_evaluate(board, player):
    # 改进模式：威胁权重更丰富
    score = 0
    lines = []
    for r in range(BOARD_SIZE):
        lines.append(board[r])
    for c in range(BOARD_SIZE):
        lines.append([board[r][c] for r in range(BOARD_SIZE)])
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        lines.append([board[r][r-d] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=r-d<BOARD_SIZE])
        lines.append([board[r][BOARD_SIZE-1-(r-d)] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=BOARD_SIZE-1-(r-d)<BOARD_SIZE])
    def pattern_score(s, player):
        patterns = {
            f"{player}{player}{player}{player}{player}": 1000000,
            f"0{player}{player}{player}{player}0": 10000,
            f"{player}{player}{player}{player}0": 5000,
            f"0{player}{player}{player}{player}": 5000,
            f"0{player}{player}{player}0": 1000,  # 提升活三权重
            f"0{player}{player}0": 200,          # 提升活二权重
            f"0{player}{player}{player}{player}1": 800, # 冲四
            f"1{player}{player}{player}{player}0": 800, # 冲四
            f"0{player}{player}{player}00": 120, # 眠三
            f"00{player}{player}{player}0": 120,
            f"0{player}{player}0{player}0": 120,
            f"0{player}0{player}{player}0": 120,
        }
        scr = 0
        for pat, val in patterns.items():
            scr += s.count(pat) * val
        return scr
    for line in lines:
        s = ''.join(str(x) for x in line)
        score += pattern_score(s, player)
        score -= pattern_score(s, other(player)) * 1.3
    return score

#########################################
# 用于Easy的更完整权重
def influence_map(board, player):
    inf_map = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c]!=0: continue
            for dr in range(-2,3):
                for dc in range(-2,3):
                    nr, nc = r+dr, c+dc
                    if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE:
                        if board[nr][nc]==player:
                            inf_map[r][c]+=3 if abs(dr)+abs(dc)==1 else 2
                        elif board[nr][nc]==other(player):
                            inf_map[r][c]-=2 if abs(dr)+abs(dc)==1 else 1
    return inf_map

def custom_priority_algo(board, player=AI):
    log("Using Custom Priority Algorithm")
    moves = get_moves(board)
    # 1. 立即取胜
    for move in moves:
        b = move_board(board, move, player)
        if check_win(b, player):
            log(f"Custom: Found immediate win at {move}")
            return move
    # 2. 立即防守
    for move in moves:
        b = move_board(board, move, other(player))
        if check_win(b, other(player)):
            log(f"Custom: Blocking opponent win at {move}")
            return move
    # 3. 检查冲四等威胁
    for move in moves:
        b = move_board(board, move, other(player))
        h3, h4, d3, d4, rush4, sleep3 = threat_analysis(b, other(player))
        if h4 > 0 or d3 or d4 or rush4 > 0:
            log(f"Custom: Blocking threat at {move}")
            return move
    # 4. 中心优先（中心被占后寻找次中心点）
    centers = [(BOARD_SIZE//2, BOARD_SIZE//2)]
    # 次中心点
    if board[centers[0][0]][centers[0][1]] != 0:
        for dr in range(-1,2):
            for dc in range(-1,2):
                nr, nc = BOARD_SIZE//2+dr, BOARD_SIZE//2+dc
                if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==0:
                    centers.append((nr,nc))
    for c in centers:
        if board[c[0]][c[1]]==0:
            log(f"Custom: Taking center or near center {c}")
            return c
    # 5. 评分
    best_score = -float('inf')
    best_move = moves[0]
    inf_map = influence_map(board, player)
    for move in moves:
        b = move_board(board, move, player)
        my_h3, my_h4, my_d3, my_d4, my_rush4, my_sleep3 = threat_analysis(b, player)
        op_h3, op_h4, op_d3, op_d4, op_rush4, op_sleep3 = threat_analysis(b, other(player))
        s = 0
        s += my_h4 * 9000 + my_d4 * 13000 + my_h3 * 800 + my_d3 * 1300 + my_rush4 * 900 + my_sleep3 * 200
        s -= op_h4 * 9500 + op_d4 * 14000 + op_h3 * 900 + op_d3 * 1500 + op_rush4*1200 + op_sleep3*300
        s += inf_map[move[0]][move[1]] * 3
        if s > best_score:
            best_score = s
            best_move = move
    log(f"Custom: Best move by weighted eval is {best_move} score {best_score}")
    return best_move

#########################################
def simulated_annealing(board, player, eval_func, max_iter=500, T0=1.0, Tf=0.01):
    log("Using Simulated Annealing")
    moves = get_moves(board)
    if not moves: return moves[0]
    curr_move = random.choice(moves)
    curr_score = eval_func(move_board(board, curr_move, player), AI)
    T = T0
    best_move, best_score = curr_move, curr_score
    for i in range(max_iter):
        next_move = random.choice(moves)
        temp_board = move_board(board, next_move, player)
        next_score = eval_func(temp_board, AI)
        if next_score > curr_score or math.exp((next_score-curr_score)/T) > random.random():
            curr_move, curr_score = next_move, next_score
        if curr_score > best_score:
            best_move, best_score = curr_move, curr_score
        T = T0 * (Tf / T0) ** (i/max_iter)
        if T < Tf:
            break
    return best_move

def pattern_exploit_algo(board, player=AI):
    log("Using Pattern Exploit Algorithm")
    moves = get_moves(board)
    for move in moves:
        b = move_board(board, move, other(player))
        for next_move in get_moves(b):
            bb = move_board(b, next_move, player)
            if check_win(bb, player):
                log(f"PatternExploit: Found kill at {next_move}")
                return next_move
    for move in moves:
        b = move_board(board, move, other(player))
        h3, h4, _, _, rush4, _ = threat_analysis(b, player)
        if h4 > 0 or h3 > 1 or rush4 > 0:
            log(f"PatternExploit: Forcing threat at {move}")
            return move
    return custom_priority_algo(board, player)

def nn_heuristic(board, player=AI):
    log("Using Neural Net Heuristic (simulated)")
    moves = get_moves(board)
    best = moves[0]
    best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player) + random.randint(-30,30)
        h3, h4, d3, d4, rush4, sleep3 = threat_analysis(b, player)
        s += (h3*250 + h4*900 + d3*1800 + d4*3500 + rush4*1000 + sleep3*300)
        if s > best_score:
            best, best_score = (r,c), s
    log(f"NNHeuristic: Best move {best} score {best_score}")
    return best

def is_serious(board):
    for player in [AI, HUMAN]:
        if check_win(board, player):
            return True
    for player in [AI, HUMAN]:
        h3, h4, d3, d4, rush4, sleep3 = threat_analysis(board, player)
        if h4 >= 1 or d3 or d4 or rush4 > 0:
            return True
    for player in [AI, HUMAN]:
        h3, _, _, _, _, _ = threat_analysis(board, player)
        if h3 >= 3:
            return True
    return False

#########################################
# 三阶启发式排序与表
class HeuristicTable:
    def __init__(self):
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
    def update_killer(self, depth, move):
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth] = self.killer_moves[depth][:2]
    def update_history(self, move, depth):
        key = move
        self.history_table[key] += 2 ** depth
    def get_killer(self, depth):
        return self.killer_moves[depth]
    def get_history(self, move):
        return self.history_table[move]

def threat_points(board, player):
    must_moves = []
    for move in get_moves(board):
        b1 = move_board(board, move, player)
        if check_win(b1, player):
            must_moves.append(move)
        b2 = move_board(board, move, other(player))
        if check_win(b2, other(player)):
            must_moves.append(move)
    return list(set(must_moves))

def heuristic_sort(board, moves, player, heur_table=None, depth=0):
    threat = set(threat_points(board, player))
    killer = set(heur_table.get_killer(depth)) if heur_table else set()
    scored = []
    for m in moves:
        score = 0
        if m in threat: score += 10000
        if m in killer: score += 5000
        if heur_table: score += heur_table.get_history(m)
        score += random.randint(0, 10)
        scored.append((score, m))
    scored.sort(reverse=True)
    return [m for _, m in scored]

def pvs_search(board, player, depth, alpha, beta, eval_func, heur_table, max_depth, killer_moves, result_queue, stop_event, adaptive=False):
    def _pvs(bd, ply, d, a, b, depth0):
        if stop_event and stop_event.is_set():
            return None, 0
        if check_win(bd, AI):
            return None, 1000000 + d
        if check_win(bd, HUMAN):
            return None, -1000000 - d
        if d == 0 or board_full(bd):
            return None, eval_func(bd, AI)
        moves = get_moves(bd)
        moves = heuristic_sort(bd, moves, ply, heur_table, depth0)
        best_move = None
        first = True
        for m in moves:
            bd[m[0]][m[1]] = ply
            if first:
                _, sc = _pvs(bd, other(ply), d-1, -b, -a, depth0+1)
                sc = -sc
                first = False
            else:
                _, sc = _pvs(bd, other(ply), d-1, -a-1, -a, depth0+1)
                sc = -sc
                if a < sc < b:
                    _, sc = _pvs(bd, other(ply), d-1, -b, -sc, depth0+1)
                    sc = -sc
            bd[m[0]][m[1]] = 0
            if sc > a:
                a = sc
                best_move = m
                heur_table.update_killer(depth0, m)
                heur_table.update_history(m, depth0)
            if a >= b:
                killer_moves.append(m)
                break
        return best_move, a

    d = depth
    if adaptive:
        for playerx in [AI, HUMAN]:
            h3, h4, d3, d4, rush4, sleep3 = threat_analysis(board, playerx)
            if h4 or d3 or d4 or rush4 > 0:
                d = min(max_depth, depth + 2)
    move, score = _pvs(deepcopy(board), player, d, alpha, beta, 0)
    result_queue.put((move, score))

#########################################
class MCTSNode:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.lock = threading.Lock()
        self.virtual_loss = 0
    def ucb_score(self, total_sim, c=1.4):
        if self.visits == 0: return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(total_sim+1) / (self.visits+1)) - self.virtual_loss

class SharedMCTSTree:
    def __init__(self, board, player):
        self.root = MCTSNode()
        self.board = deepcopy(board)
        self.player = player
        self.total_sim = 0
    def select(self, node):
        current = node
        path = []
        while current.children:
            with current.lock:
                scores = [child.ucb_score(self.total_sim) for child in current.children]
                best_idx = scores.index(max(scores))
                current = current.children[best_idx]
            path.append(current)
        return current, path
    def expand(self, node, board, player):
        moves = get_moves(board)
        for m in moves:
            child = MCTSNode(m, parent=node)
            node.children.append(child)
    def simulate(self, board, player):
        curr = player
        b = deepcopy(board)
        while not check_win(b, AI) and not check_win(b, HUMAN) and not board_full(b):
            ms = get_moves(b)
            if not ms: break
            rr, cc = random.choice(ms)
            b[rr][cc] = curr
            curr = other(curr)
        if check_win(b, AI):
            return 1
        elif check_win(b, HUMAN):
            return -1
        else:
            return 0
    def backpropagate(self, path, result):
        for node in path:
            with node.lock:
                node.visits += 1
                if result == 1:
                    node.wins += 1
                elif result == -1:
                    node.wins -= 1
    def run_simulation(self, sim_time=2.5, max_sims=300, stop_event=None):
        # 更高模拟次数，专业模式建议300次
        start = time.time()
        sim_count = 0
        while sim_count < max_sims and (time.time() - start) < sim_time:
            if stop_event and stop_event.is_set():
                break
            current, path = self.select(self.root)
            b = deepcopy(self.board)
            node = self.root
            for pnode in path:
                if pnode.move:
                    b[pnode.move[0]][pnode.move[1]] = self.player
            if not current.children:
                self.expand(current, b, self.player)
            if current.children:
                next_node = random.choice(current.children)
                b2 = move_board(b, next_node.move, self.player)
                with next_node.lock:
                    next_node.virtual_loss += 1
                result = self.simulate(b2, self.player)
                with next_node.lock:
                    next_node.virtual_loss -= 1
                self.backpropagate([current, next_node], result)
            else:
                result = self.simulate(b, self.player)
                self.backpropagate([current], result)
            self.total_sim += 1
            sim_count += 1
    def best_move(self):
        if not self.root.children:
            return random.choice(get_moves(self.board))
        visits = [child.visits for child in self.root.children]
        idx = visits.index(max(visits))
        return self.root.children[idx].move

#########################################
# ============ Professional duel (Beta) 全新优化 ==============
def ai_move_pro(board, player=AI):
    log("Mode: Professional duel (Beta) [Fast Deep MCTS+PVS]")
    if is_serious(board):
        pe = pattern_exploit_algo(board, player)
        if pe: return pe
        num_threads = min(multiprocessing.cpu_count(), 14)
        heur_table = HeuristicTable()
        # ------- 线程池资源均衡分配 ----------
        moves = get_moves(board)
        # 取前num_threads+2个候选，提高多样性，线程池分担
        move_candidates = heuristic_sort(board, moves, player, heur_table, 0)[:num_threads+2]
        killer_moves = []
        result_q = Queue()
        def mcts_pvs_worker(move, depth, heur_table, result_q, killer_moves):
            b = move_board(board, move, player)
            # MCTS模拟次数提升
            mcts_tree = SharedMCTSTree(b, other(player))
            mcts_tree.run_simulation(sim_time=2.2, max_sims=300)
            mcts_best = mcts_tree.best_move()
            # PVS深度提升至6-9层，且根据局势自适应
            adaptive = is_serious(b)
            result_queue = Queue()
            pvs_search(
                move_board(b, mcts_best, other(player)), player, 8 if adaptive else 6,
                -float('inf'), float('inf'),
                heuristic_evaluate, heur_table, 9, killer_moves, result_queue, threading.Event(), adaptive)
            if not result_queue.empty():
                next_move, score = result_queue.get()
                # 总分 = MCTS访问数+PVS分数加权
                score += mcts_tree.root.visits * 7
                result_q.put(((move, mcts_best, next_move), score))
            else:
                result_q.put(((move, mcts_best, None), -999999))
        # 线程池分配
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = []
            for idx, mov in enumerate(move_candidates):
                depth = 6 + (1 if idx == 0 else 0)
                futures.append(pool.submit(mcts_pvs_worker, mov, depth, heur_table, result_q, killer_moves))
            start_time = time.time()
            best_score = -float('inf')
            best_move = move_candidates[0]
            best_path = None
            while time.time() - start_time < TIME_LIMIT-2:
                if not result_q.empty():
                    (mv, score) = result_q.get()
                    if score > best_score:
                        best_score = score
                        best_move = mv[0]
                        best_path = mv
                # 资源均衡等待所有线程结束
                if all(f.done() for f in futures):
                    break
                time.sleep(0.01)
        log(f"Pro mode deep MCTS+PVS candidates: {move_candidates}, best:{best_move} path:{best_path} score:{best_score}")
        return best_move
    else:
        # 更丰富的投票机制，所有算法权重加权
        m1 = nn_heuristic(board, player)
        m2 = simulated_annealing(board, player, heuristic_evaluate)
        m3 = custom_priority_algo(board, player)
        m4 = pattern_exploit_algo(board, player)
        # 结合MCTS
        m5 = mcts(board, player, sim_time=1.5, min_sims=100, max_sims=150)
        results = [m1, m2, m2, m3, m3, m4, m5] # m2, m3加权
        count = Counter(results)
        most_common = count.most_common(1)[0][0]
        log(f"Pro mode (not serious) vote: {results} => {most_common}")
        return most_common

#########################################
def minimax(board, player, depth, eval_func):
    log("Using Minimax. depth=%d" % depth)
    winner = check_win(board, AI)
    loser  = check_win(board, HUMAN)
    if winner:
        return None, 1000000
    if loser:
        return None, -1000000
    if depth == 0 or board_full(board):
        return None, eval_func(board, AI)
    best_move = None
    if player == AI:
        best_score = -float('inf')
        for r, c in get_moves(board):
            board[r][c] = player
            _, score = minimax(board, other(player), depth-1, eval_func)
            board[r][c] = 0
            if score > best_score:
                best_score = score
                best_move = (r, c)
        log(f"Minimax best move: {best_move} score: {best_score}")
        return best_move, best_score
    else:
        best_score = float('inf')
        for r, c in get_moves(board):
            board[r][c] = player
            _, score = minimax(board, other(player), depth-1, eval_func)
            board[r][c] = 0
            if score < best_score:
                best_score = score
                best_move = (r, c)
        return best_move, best_score

def alpha_beta(board, player, depth, alpha, beta, eval_func):
    log("Using Alpha-Beta. depth=%d" % depth)
    winner = check_win(board, AI)
    loser  = check_win(board, HUMAN)
    if winner:
        return None, 1000000
    if loser:
        return None, -1000000
    if depth==0 or board_full(board):
        return None, eval_func(board, AI)
    best_move = None
    moves = sorted(get_moves(board), key=lambda m: -eval_func(move_board(board, m, player), AI))
    if player == AI:
        value = -float('inf')
        for r,c in moves:
            board[r][c]=player
            _, score = alpha_beta(board, other(player), depth-1, alpha, beta, eval_func)
            board[r][c]=0
            if score > value:
                value = score
                best_move = (r,c)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        log(f"AlphaBeta best move: {best_move} score: {value}")
        return best_move, value
    else:
        value = float('inf')
        for r,c in moves:
            board[r][c]=player
            _, score = alpha_beta(board, other(player), depth-1, alpha, beta, eval_func)
            board[r][c]=0
            if score < value:
                value = score
                best_move = (r,c)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_move, value

def mcts(board, player, sim_time=1.5, min_sims=80, max_sims=150):
    log(f"Using MCTS, sim_time={sim_time}, min_sims={min_sims}, max_sims={max_sims}")
    start = time.time()
    moves = get_moves(board)
    best_move = moves[0]
    best_score = -1
    sims_done = 0
    for r,c in moves:
        wins = 0
        plays = 0
        for _ in range(max_sims):
            b = deepcopy(board)
            b[r][c]=player
            curr = other(player)
            while not check_win(b,AI) and not check_win(b,HUMAN) and not board_full(b):
                ms = get_moves(b)
                if not ms: break
                rr,cc = random.choice(ms)
                b[rr][cc]=curr
                curr = other(curr)
            if check_win(b, AI):
                wins += 1
            plays += 1
            sims_done += 1
            if sims_done >= min_sims and time.time() - start > sim_time:
                break
        log(f"[MCTS] Move {r},{c} - win {wins} / {plays}")
        if wins > best_score:
            best_score = wins
            best_move = (r, c)
        if sims_done >= min_sims and time.time()-start > sim_time:
            break
    log(f"[MCTS] Simulations done: {sims_done}")
    return best_move

#########################################
# =============== Hell模式多进程优化 =================
def ai_move_hell(board, player=AI):
    log("Mode: Hell")
    if is_serious(board):
        moveset = []
        # 进程池超时处理和进程间共享数据
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(minimax, deepcopy(board), player, 3, heuristic_evaluate),
                executor.submit(alpha_beta, deepcopy(board), player, 4, -float('inf'), float('inf'), heuristic_evaluate),
                executor.submit(mcts, deepcopy(board), player, 1.6, 120, 180),
                executor.submit(simulated_annealing, deepcopy(board), player, heuristic_evaluate),
                executor.submit(custom_priority_algo, deepcopy(board), player),
            ]
            moves = get_moves(board)
            best = moves[0]
            best_score = -float('inf')
            for r, c in moves:
                b = move_board(board, (r, c), player)
                s = heuristic_evaluate(b, player)
                if s > best_score:
                    best, best_score = (r, c), s
            moveset.append(best)
            done_count = 0
            for future in as_completed(futures, timeout=TIME_LIMIT-2):
                try:
                    res = future.result(timeout=TIME_LIMIT-2)
                    if isinstance(res, tuple): moveset.append(res[0])
                    else: moveset.append(res)
                except Exception:
                    continue
                done_count += 1
            # 若进程超时，仍能投票
            count = Counter(moveset)
            most_common = count.most_common(1)[0][0]
            log(f"Hell mode vote: {moveset} => {most_common} done: {done_count}")
            return most_common
    else:
        # 非严重局面依然采纳所有投票，降级但不降为difficult
        m1 = nn_heuristic(board, player)
        m2 = simulated_annealing(board, player, heuristic_evaluate)
        m3 = custom_priority_algo(board, player)
        m4 = mcts(board, player, sim_time=1.2, min_sims=80, max_sims=120)
        results = [m1, m2, m2, m3, m4] # m2加权
        count = Counter(results)
        most_common = count.most_common(1)[0][0]
        log(f"Hell mode (not serious) vote: {results} => {most_common}")
        return most_common

#########################################
def ai_move_easy(board, player=AI):
    log("Mode: Easy")
    return custom_priority_algo(board, player)

def ai_move_normal(board, player=AI):
    log("Mode: Normal")
    m1 = simulated_annealing(board, player, heuristic_evaluate)
    moves = get_moves(board)
    best = moves[0]
    best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player)
        if s > best_score:
            best, best_score = (r, c), s
    m2 = best
    m3 = custom_priority_algo(board, player)
    # 改进：投票机制加权，启发式更高权重
    results = [m1, m2, m2, m3]
    count = Counter(results)
    most_common = count.most_common(1)[0][0]
    log(f"Normal mode vote: {results} => {most_common}")
    return most_common

def ai_move_difficult(board, player=AI):
    log("Mode: Difficult")
    m1 = simulated_annealing(board, player, heuristic_evaluate)
    m2 = mcts(board, player, sim_time=1.2, min_sims=80, max_sims=120)
    moves = get_moves(board)
    best = moves[0]
    best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player)
        if s > best_score:
            best, best_score = (r, c), s
    m3 = best
    m4 = custom_priority_algo(board, player)
    # 改进：投票机制加权
    results = [m1, m2, m2, m3, m4]
    count = Counter(results)
    most_common = count.most_common(1)[0][0]
    log(f"Difficult mode vote: {results} => {most_common}")
    return most_common

#########################################
def ai_thread_func(board, result_queue, timer_queue, mode_ai_func, player, stop_event=None):
    move = None
    def ai_job():
        nonlocal move
        move = mode_ai_func(board, player)
        if not stop_event or not stop_event.is_set():
            result_queue.put(move)
    t = threading.Thread(target=ai_job)
    t.start()
    t.join(TIME_LIMIT-1)
    if move is None and (not stop_event or not stop_event.is_set()):
        move = custom_priority_algo(board, player)
        result_queue.put(move)
    timer_queue.put('done')

def countdown_thread_func(secs, cd_queue, stop_event=None):
    for i in range(secs, -1, -1):
        if stop_event and stop_event.is_set():
            break
        cd_queue.put(i)
        time.sleep(1)

def reset_game():
    return [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)], HUMAN, None, False

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE_X, SCREEN_SIZE_Y))
    pygame.display.set_caption("Gomoku AI-https://github.com/my-txz/AI- ")
    clock = pygame.time.Clock()
    board, turn, winner, ai_thinking = reset_game()
    ai_result_queue = Queue()
    ai_timer_queue = Queue()
    ai_cd_queue = Queue()
    ai_thread = None
    cd_thread = None
    font = pygame.font.SysFont(None, 22)
    replay_btn_rect = None
    stop_ai_duel_btn_rect = None
    mode_btn_rects = []
    ai_cd = 0
    mode_idx = 1
    ai_vs_ai = False
    ai_vs_ai_turn = AI
    stop_duel_event = threading.Event()

    mode_ai_map = {
        "easy": ai_move_easy,
        "normal": ai_move_normal,
        "difficult": ai_move_difficult,
        "hell": ai_move_hell,
    }
    if BETA_MODE:
        mode_ai_map["pro"] = ai_move_pro
        mode_ai_map["ai_vs_ai"] = ai_move_hell

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if replay_btn_rect and replay_btn_rect.collidepoint(mx, my):
                    board, turn, winner, ai_thinking = reset_game()
                    ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue()
                    ai_thread = None; cd_thread = None; ai_cd = 0; ai_vs_ai = (BETA_MODE and MODES[mode_idx][1] == "ai_vs_ai")
                    ai_vs_ai_turn = AI
                    stop_duel_event.clear()
                if stop_ai_duel_btn_rect and stop_ai_duel_btn_rect.collidepoint(mx, my):
                    stop_duel_event.set()
                    ai_vs_ai = False
                    winner = "Stopped"
                    ai_thinking = False
                for idx, btn in enumerate(mode_btn_rects):
                    if btn.collidepoint(mx, my):
                        mode_idx = idx
                        board, turn, winner, ai_thinking = reset_game()
                        ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue()
                        ai_thread = None; cd_thread = None; ai_cd = 0
                        ai_vs_ai = (BETA_MODE and MODES[mode_idx][1] == "ai_vs_ai")
                        ai_vs_ai_turn = AI
                        stop_duel_event.clear()
                if not ai_vs_ai and turn==HUMAN and not winner and not ai_thinking:
                    offset_y = TOP_PANEL_HEIGHT
                    if (MARGIN-CELL_SIZE//2 <= mx <= MARGIN + CELL_SIZE * (BOARD_SIZE-1) + CELL_SIZE//2 and
                        offset_y + MARGIN-CELL_SIZE//2 <= my <= offset_y + MARGIN + CELL_SIZE * (BOARD_SIZE-1) + CELL_SIZE//2):
                        col = int(round((mx-MARGIN)/CELL_SIZE))
                        row = int(round((my-offset_y-MARGIN)/CELL_SIZE))
                        if 0<=row<BOARD_SIZE and 0<=col<BOARD_SIZE and board[row][col]==0:
                            board[row][col]=HUMAN
                            if check_win(board, HUMAN):
                                winner = 'Human'
                            elif board_full(board):
                                winner = 'Draw'
                            else:
                                turn = AI
                                ai_thinking = True
                                ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue()
                                ai_thread = threading.Thread(target=ai_thread_func, args=(
                                    deepcopy(board), ai_result_queue, ai_timer_queue, mode_ai_map[MODES[mode_idx][1]], AI))
                                ai_thread.start()
                                cd_thread = threading.Thread(target=countdown_thread_func, args=(TIME_LIMIT, ai_cd_queue))
                                cd_thread.daemon = True
                                cd_thread.start()

        if ai_vs_ai and not ai_thinking and not winner and not stop_duel_event.is_set():
            ai_thinking = True
            ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue()
            player = ai_vs_ai_turn
            ai_thread = threading.Thread(target=ai_thread_func, args=(
                deepcopy(board), ai_result_queue, ai_timer_queue, mode_ai_map["ai_vs_ai"], player, stop_duel_event))
            ai_thread.start()
            cd_thread = threading.Thread(target=countdown_thread_func, args=(TIME_LIMIT, ai_cd_queue, stop_duel_event))
            cd_thread.daemon = True
            cd_thread.start()

        if (turn==AI or ai_vs_ai) and not winner and ai_thinking:
            if not ai_cd_queue.empty():
                ai_cd = ai_cd_queue.get()
            else:
                ai_cd = 0
            if not ai_result_queue.empty():
                move = ai_result_queue.get()
                player = AI if not ai_vs_ai else ai_vs_ai_turn
                if move:
                    board[move[0]][move[1]] = player
                    if check_win(board, player):
                        winner = f"{'AI1' if ai_vs_ai and player==AI else ('AI2' if ai_vs_ai and player==HUMAN else 'AI')}"
                    elif board_full(board):
                        winner = 'Draw'
                    else:
                        if ai_vs_ai:
                            ai_vs_ai_turn = other(ai_vs_ai_turn)
                        else:
                            turn = HUMAN
                ai_thinking = False

        # Draw GUI
        screen.fill(BG_COLOR)
        panel_y = 10
        time_color = RED if ai_thinking else (50, 120, 80)
        font_big = pygame.font.SysFont(None, 28)
        txt_time = font_big.render(f"Time left: {ai_cd}s", True, time_color)
        screen.blit(txt_time, (MARGIN, panel_y))
        panel_y += 32
        if winner:
            txt_status = font_big.render(f"Status: {winner}", True, RED)
        else:
            txt_status = font_big.render(f"Status: Playing", True, (50, 120, 80))
        screen.blit(txt_status, (MARGIN, panel_y))

        draw_board(screen, board)
        y_panel = TOP_PANEL_HEIGHT + CELL_SIZE * BOARD_SIZE + MARGIN + 10
        mode_btn_rects = draw_mode_buttons(screen, font, mode_idx, y_panel)
        if BETA_MODE and ai_vs_ai:
            stop_ai_duel_btn_rect = draw_stop_ai_duel_button(screen, font, y_panel + BTN_HEIGHT * 2 + BTN_SPACING * 2)
            replay_btn_rect = None
        else:
            stop_ai_duel_btn_rect = None
            replay_btn_rect = draw_replay_button(screen, font, y_panel + BTN_HEIGHT * 2 + BTN_SPACING * 2)

        info_x = MARGIN
        info_y = y_panel + BTN_HEIGHT * 2 + BTN_SPACING * 2 + BTN_HEIGHT + 10
        mode_label = MODES[mode_idx][0]
        txtMode = font.render(f"Mode: {mode_label}", True, RED)
        screen.blit(txtMode, (info_x, info_y))
        info_y += 24
        if ai_thinking:
            txt1 = font.render(f"AI is thinking...", True, RED)
            screen.blit(txt1, (info_x, info_y))
            info_y += 24
        h3, h4, d3, d4, rush4, sleep3 = threat_analysis(board, AI)
        txt3 = font.render(f"AI open-3s: {h3} 4s: {h4} double-3: {d3} double-4: {d4} rush4: {rush4} sleep3: {sleep3}", True, RED)
        screen.blit(txt3, (info_x, info_y))
        if BETA_MODE:
            info_y += 24
            txtB = font.render("BETA: CPU AND GPU Accelerate, Pro Duel, AI VS AI", True, (0,128,255))
            screen.blit(txtB, (info_x, info_y))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

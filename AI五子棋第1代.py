import pygame
import sys
import math
import random
import time
import threading
from queue import Queue
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==== 基础参数 ====
BOARD_SIZE = 15
CELL_SIZE = 40
MARGIN = 40
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE + MARGIN * 2 + 140  # Extra space for buttons/info
FPS = 30
TIME_LIMIT = 60  # seconds

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BG_COLOR = (200, 170, 120)
BTN_COLOR = (90, 160, 220)
BTN_COLOR2 = (100, 100, 100)

HUMAN = 1
AI = 2

MODES = [
    ("Easy", "easy"),
    ("Normal", "normal"),
    ("Difficult", "difficult"),
    ("Hell", "hell"),
    ("Professional duel (Beta)", "pro"),
    ("AI VS AI (Beta)", "ai_vs_ai"),
]

def other(player):
    return HUMAN if player == AI else AI

def draw_board(screen, board):
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (MARGIN + CELL_SIZE * i, MARGIN), (MARGIN + CELL_SIZE * i, MARGIN + CELL_SIZE * (BOARD_SIZE-1)), 1)
        pygame.draw.line(screen, BLACK, (MARGIN, MARGIN + CELL_SIZE * i), (MARGIN + CELL_SIZE * (BOARD_SIZE-1), MARGIN + CELL_SIZE * i), 1)
    for i in [3, 7, 11]:
        for j in [3, 7, 11]:
            pygame.draw.circle(screen, BLACK, (MARGIN + CELL_SIZE * i, MARGIN + CELL_SIZE * j), 5)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == HUMAN:
                pygame.draw.circle(screen, BLACK, (MARGIN + CELL_SIZE * c, MARGIN + CELL_SIZE * r), CELL_SIZE // 2 - 2)
            elif board[r][c] == AI:
                pygame.draw.circle(screen, WHITE, (MARGIN + CELL_SIZE * c, MARGIN + CELL_SIZE * r), CELL_SIZE // 2 - 2)
                pygame.draw.circle(screen, BLACK, (MARGIN + CELL_SIZE * c, MARGIN + CELL_SIZE * r), CELL_SIZE // 2 - 2, 1)

def draw_replay_button(screen, font):
    btn_rect = pygame.Rect(SCREEN_SIZE-200, SCREEN_SIZE-100, 140, 40)
    pygame.draw.rect(screen, BTN_COLOR, btn_rect, border_radius=8)
    txt = font.render("Replay", True, (255,255,255))
    screen.blit(txt, (SCREEN_SIZE-170, SCREEN_SIZE-90))
    return btn_rect

def draw_mode_buttons(screen, font, current_idx):
    btns = []
    btn_width = 220
    btn_height = 35
    btn_spacing = 12
    base_y = SCREEN_SIZE - 100
    base_x = 10
    for idx, (label, _) in enumerate(MODES):
        btn_rect = pygame.Rect(base_x + (btn_width + btn_spacing) * (idx % 2), base_y + (btn_height + 8) * (idx // 2), btn_width, btn_height)
        c = (BTN_COLOR if idx!=current_idx else (220,60,60))
        pygame.draw.rect(screen, c, btn_rect, border_radius=8)
        txt = font.render(label, True, (255,255,255))
        screen.blit(txt, (btn_rect.x+10, btn_rect.y+7))
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

def minimax(board, player, depth, eval_func):
    print("AI uses: Minimax")
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
    print("AI uses: Alpha-Beta Pruning")
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

def mcts(board, player, sim_time=1.5, n_sim=120):
    print(f"AI uses: MCTS ({n_sim} simulations)")
    start = time.time()
    moves = get_moves(board)
    best_move = moves[0]
    best_score = -1
    for r,c in moves:
        wins = 0
        plays = 0
        for _ in range(n_sim):
            b = deepcopy(board)
            b[r][c]=player
            curr = other(player)
            step = 0
            while not check_win(b,AI) and not check_win(b,HUMAN) and not board_full(b) and step < 200:
                ms = get_moves(b)
                if not ms: break
                rr,cc = random.choice(ms)
                b[rr][cc]=curr
                curr = other(curr)
                step += 1
            if check_win(b, AI):
                wins += 1
            plays += 1
        if wins > best_score:
            best_score = wins
            best_move = (r, c)
        if time.time()-start > sim_time:
            break
    return best_move

def simulated_annealing(board, player, eval_func, max_iter=400, T0=1.0, Tf=0.01):
    print("AI uses: Simulated Annealing")
    moves = get_moves(board)
    if not moves: return moves[0]
    curr_move = random.choice(moves)
    curr_score = eval_func(move_board(board, curr_move, player), AI)
    T = T0
    for i in range(max_iter):
        next_move = random.choice(moves)
        temp_board = move_board(board, next_move, player)
        next_score = eval_func(temp_board, AI)
        if next_score > curr_score or math.exp((next_score-curr_score)/T) > random.random():
            curr_move, curr_score = next_move, next_score
        T = T0 * (Tf / T0) ** (i/max_iter)
        if T < Tf:
            break
    return curr_move

def heuristic_evaluate(board, player):
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
            f"0{player}{player}{player}0": 800,
            f"0{player}{player}0": 150,
        }
        for pat, val in patterns.items():
            score = s.count(pat) * val
            if score > 0:
                return score
        return 0
    for line in lines:
        s = ''.join(str(x) for x in line)
        score += pattern_score(s, player)
        score -= pattern_score(s, other(player)) * 1.2
    return score

def threat_analysis(board, player):
    lines = []
    for r in range(BOARD_SIZE):
        lines.append(board[r])
    for c in range(BOARD_SIZE):
        lines.append([board[r][c] for r in range(BOARD_SIZE)])
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        lines.append([board[r][r-d] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=r-d<BOARD_SIZE])
        lines.append([board[r][BOARD_SIZE-1-(r-d)] for r in range(max(0,d), min(BOARD_SIZE,BOARD_SIZE+d)) if 0<=BOARD_SIZE-1-(r-d)<BOARD_SIZE])
    h3, h4, d3, d4 = 0, 0, 0, 0
    threes = 0
    fours = 0
    for line in lines:
        s = ''.join(str(x) for x in line)
        if f'0{player}{player}{player}0' in s:
            h3 += 1
            threes += 1
        if f'0{player}{player}{player}{player}0' in s:
            h4 += 1
            fours += 1
    if threes >= 2: d3 = 1
    if fours >= 2: d4 = 1
    return h3, h4, d3, d4

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
                            inf_map[r][c]+=2
                        elif board[nr][nc]==other(player):
                            inf_map[r][c]-=1
    return inf_map

def custom_priority_algo(board, player=AI):
    print("AI uses: Improved Priority Algorithm")
    moves = get_moves(board)
    # 1. Win immediately
    for move in moves:
        b = move_board(board, move, player)
        if check_win(b, player):
            return move
    # 2. Block opponent win
    for move in moves:
        b = move_board(board, move, other(player))
        if check_win(b, other(player)):
            return move
    # 3. Block major threat
    for move in moves:
        b = move_board(board, move, other(player))
        h3, h4, d3, d4 = threat_analysis(b, other(player))
        if h4 > 0 or d3 or d4:
            return move
    # 4. Take center if empty
    center = (BOARD_SIZE//2, BOARD_SIZE//2)
    if board[center[0]][center[1]] == 0:
        return center
    # 5. Prefer next to own/opp long lines
    best_score = -float('inf')
    best_move = moves[0]
    for move in moves:
        b = move_board(board, move, player)
        my_h3, my_h4, my_d3, my_d4 = threat_analysis(b, player)
        op_h3, op_h4, op_d3, op_d4 = threat_analysis(b, other(player))
        s = 0
        s += my_h4 * 9000 + my_d4 * 12000 + my_h3 * 700 + my_d3 * 1100
        s -= op_h4 * 9500 + op_d4 * 13000 + op_h3 * 800 + op_d3 * 1200
        inf_map = influence_map(board, player)
        s += inf_map[move[0]][move[1]] * 2
        if s > best_score:
            best_score = s
            best_move = move
    return best_move

def pattern_exploit_algo(board, player=AI):
    print("AI uses: Pattern Exploit Algorithm")
    moves = get_moves(board)
    for move in moves:
        b = move_board(board, move, other(player))
        for next_move in get_moves(b):
            bb = move_board(b, next_move, player)
            if check_win(bb, player):
                return next_move
    for move in moves:
        b = move_board(board, move, other(player))
        h3, h4, _, _ = threat_analysis(b, player)
        if h4 > 0 or h3 > 1:
            return move
    return custom_priority_algo(board, player)

def nn_heuristic(board, player=AI):
    print("AI uses: Neural Net Heuristic (Simulated)")
    moves = get_moves(board)
    best = moves[0]
    best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player) + random.randint(-50,50)
        h3, h4, d3, d4 = threat_analysis(b, player)
        s += (h3*200 + h4*800 + d3*1500 + d4*3000)
        if s > best_score:
            best, best_score = (r,c), s
    return best

def is_serious(board):
    for player in [AI, HUMAN]:
        if check_win(board, player):
            return True
    for player in [AI, HUMAN]:
        h3, h4, d3, d4 = threat_analysis(board, player)
        if h4 >= 1 or d3 or d4:
            return True
    for player in [AI, HUMAN]:
        h3, _, _, _ = threat_analysis(board, player)
        if h3 >= 3:
            return True
    return False

def ai_move_easy(board, player=AI): return custom_priority_algo(board, player)
def ai_move_normal(board, player=AI):
    m1 = simulated_annealing(board, player, heuristic_evaluate)
    moves = get_moves(board)
    best = moves[0]; best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player)
        if s > best_score: best, best_score = (r, c), s
    m2 = best; m3 = custom_priority_algo(board, player)
    from collections import Counter
    results = [m1, m2, m3]; count = Counter(results)
    most_common = count.most_common(1)[0][0]
    return most_common
def ai_move_difficult(board, player=AI):
    m1 = simulated_annealing(board, player, heuristic_evaluate)
    m2 = mcts(board, player, 1.5, 160)
    moves = get_moves(board)
    best = moves[0]; best_score = -float('inf')
    for r, c in moves:
        b = move_board(board, (r, c), player)
        s = heuristic_evaluate(b, player)
        if s > best_score: best, best_score = (r, c), s
    m3 = best; m4 = custom_priority_algo(board, player)
    from collections import Counter
    results = [m1, m2, m3, m4]; count = Counter(results)
    most_common = count.most_common(1)[0][0]
    return most_common
def ai_move_hell(board, player=AI):
    if is_serious(board):
        moveset = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(minimax, deepcopy(board), player, 2, heuristic_evaluate),
                executor.submit(alpha_beta, deepcopy(board), player, 3, -float('inf'), float('inf'), heuristic_evaluate),
                executor.submit(mcts, deepcopy(board), player, 2.0, 200),
                executor.submit(simulated_annealing, deepcopy(board), player, heuristic_evaluate),
                executor.submit(custom_priority_algo, deepcopy(board), player),
            ]
            moves = get_moves(board)
            best = moves[0]; best_score = -float('inf')
            for r, c in moves:
                b = move_board(board, (r, c), player)
                s = heuristic_evaluate(b, player)
                if s > best_score: best, best_score = (r, c), s
            moveset.append(best)
            from collections import Counter
            for future in as_completed(futures, timeout=TIME_LIMIT-2):
                try:
                    res = future.result(timeout=TIME_LIMIT-2)
                    if isinstance(res, tuple): moveset.append(res[0])
                    else: moveset.append(res)
                except Exception: continue
            count = Counter(moveset)
            most_common = count.most_common(1)[0][0]
            return most_common
    else:
        return ai_move_difficult(board, player)
def ai_move_pro(board, player=AI):
    if is_serious(board):
        pe = pattern_exploit_algo(board, player)
        if pe: return pe
        moveset = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(minimax, deepcopy(board), player, 2, heuristic_evaluate),
                executor.submit(alpha_beta, deepcopy(board), player, 3, -float('inf'), float('inf'), heuristic_evaluate),
                executor.submit(mcts, deepcopy(board), player, 2.0, 200),
                executor.submit(simulated_annealing, deepcopy(board), player, heuristic_evaluate),
                executor.submit(custom_priority_algo, deepcopy(board), player),
                executor.submit(pattern_exploit_algo, deepcopy(board), player),
                executor.submit(nn_heuristic, deepcopy(board), player),
            ]
            moves = get_moves(board)
            best = moves[0]; best_score = -float('inf')
            for r, c in moves:
                b = move_board(board, (r, c), player)
                s = heuristic_evaluate(b, player)
                if s > best_score: best, best_score = (r, c), s
            moveset.append(best)
            from collections import Counter
            for future in as_completed(futures, timeout=TIME_LIMIT-2):
                try:
                    res = future.result(timeout=TIME_LIMIT-2)
                    if isinstance(res, tuple): moveset.append(res[0])
                    else: moveset.append(res)
                except Exception: continue
            count = Counter(moveset)
            most_common = count.most_common(1)[0][0]
            return most_common
    else:
        m1 = nn_heuristic(board, player)
        m2 = simulated_annealing(board, player, heuristic_evaluate)
        m3 = custom_priority_algo(board, player)
        from collections import Counter
        results = [m1, m2, m3]
        count = Counter(results)
        most_common = count.most_common(1)[0][0]
        return most_common

def ai_thread_func(board, result_queue, timer_queue, mode_ai_func, player, algo_name_queue):
    move = None
    def ai_job():
        nonlocal move
        print(f"AI THINKING ({'AI1' if player==AI else 'AI2'}) with algorithm: {mode_ai_func.__name__}")
        move = mode_ai_func(board, player)
        result_queue.put(move)
        algo_name_queue.put(mode_ai_func.__name__)
    t = threading.Thread(target=ai_job)
    t.start()
    t.join(TIME_LIMIT-1)
    if move is None:
        move = custom_priority_algo(board, player)
        result_queue.put(move)
        algo_name_queue.put("custom_priority_algo (timeout)")
    timer_queue.put('done')

def countdown_thread_func(secs, cd_queue):
    for i in range(secs, -1, -1):
        cd_queue.put(i)
        time.sleep(1)

def reset_game():
    return [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)], HUMAN, None, False

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Gomoku 15x15 - Human VS AI")
    clock = pygame.time.Clock()
    board, turn, winner, ai_thinking = reset_game()
    ai_result_queue = Queue()
    ai_timer_queue = Queue()
    ai_cd_queue = Queue()
    ai_algo_queue = Queue()
    ai_thread = None
    cd_thread = None
    font = pygame.font.SysFont(None, 28)
    replay_btn_rect = None
    mode_btn_rects = []
    ai_cd = 0
    mode_idx = 1  # 默认Normal
    ai_vs_ai = False
    ai_vs_ai_turn = AI  # AI先手
    last_ai_algo = ""

    mode_ai_map = {
        "easy": ai_move_easy,
        "normal": ai_move_normal,
        "difficult": ai_move_difficult,
        "hell": ai_move_hell,
        "pro": ai_move_pro,
        "ai_vs_ai": ai_move_hell  # 可以换成ai_move_pro以体验更强对决
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if replay_btn_rect and replay_btn_rect.collidepoint(mx, my):
                    board, turn, winner, ai_thinking = reset_game()
                    ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue(); ai_algo_queue = Queue()
                    ai_thread = None; cd_thread = None; ai_cd = 0; ai_vs_ai = (MODES[mode_idx][1] == "ai_vs_ai")
                    ai_vs_ai_turn = AI; last_ai_algo = ""
                for idx, btn in enumerate(mode_btn_rects):
                    if btn.collidepoint(mx, my):
                        mode_idx = idx
                        board, turn, winner, ai_thinking = reset_game()
                        ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue(); ai_algo_queue = Queue()
                        ai_thread = None; cd_thread = None; ai_cd = 0
                        ai_vs_ai = (MODES[mode_idx][1] == "ai_vs_ai")
                        ai_vs_ai_turn = AI; last_ai_algo = ""
                if not ai_vs_ai and turn==HUMAN and not winner and not ai_thinking:
                    if (MARGIN-CELL_SIZE//2 <= mx <= SCREEN_SIZE-MARGIN+CELL_SIZE//2 and
                        MARGIN-CELL_SIZE//2 <= my <= SCREEN_SIZE-MARGIN+CELL_SIZE//2):
                        col = int(round((mx-MARGIN)/CELL_SIZE))
                        row = int(round((my-MARGIN)/CELL_SIZE))
                        if 0<=row<BOARD_SIZE and 0<=col<BOARD_SIZE and board[row][col]==0:
                            board[row][col]=HUMAN
                            if check_win(board, HUMAN):
                                winner = 'Human'
                            elif board_full(board):
                                winner = 'Draw'
                            else:
                                turn = AI
                                ai_thinking = True
                                ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue(); ai_algo_queue = Queue()
                                ai_thread = threading.Thread(target=ai_thread_func, args=(
                                    deepcopy(board), ai_result_queue, ai_timer_queue, mode_ai_map[MODES[mode_idx][1]], AI, ai_algo_queue))
                                ai_thread.start()
                                cd_thread = threading.Thread(target=countdown_thread_func, args=(TIME_LIMIT, ai_cd_queue))
                                cd_thread.daemon = True
                                cd_thread.start()

        # AI VS AI模式
        if ai_vs_ai and not ai_thinking and not winner:
            ai_thinking = True
            ai_result_queue = Queue(); ai_timer_queue = Queue(); ai_cd_queue = Queue(); ai_algo_queue = Queue()
            player = ai_vs_ai_turn
            ai_thread = threading.Thread(target=ai_thread_func, args=(
                deepcopy(board), ai_result_queue, ai_timer_queue, mode_ai_map["ai_vs_ai"], player, ai_algo_queue))
            ai_thread.start()
            cd_thread = threading.Thread(target=countdown_thread_func, args=(TIME_LIMIT, ai_cd_queue))
            cd_thread.daemon = True
            cd_thread.start()

        if (turn==AI or ai_vs_ai) and not winner and ai_thinking:
            if not ai_cd_queue.empty():
                ai_cd = ai_cd_queue.get()
            else:
                ai_cd = 0
            if not ai_result_queue.empty():
                move = ai_result_queue.get()
                if not ai_algo_queue.empty():
                    last_ai_algo = ai_algo_queue.get()
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

        draw_board(screen, board)
        replay_btn_rect = draw_replay_button(screen, font)
        mode_btn_rects = draw_mode_buttons(screen, font, mode_idx)
        mode_label = MODES[mode_idx][0]
        txtMode = font.render(f"Mode: {mode_label}", True, RED)
        screen.blit(txtMode, (10, 10))
        status_y = 48
        if ai_thinking:
            txt1 = font.render(f"AI is thinking... ({ai_cd}s left)", True, RED)
            screen.blit(txt1, (10, status_y))
            status_y += 28
        if last_ai_algo:
            txt_algo = font.render(f"AI Last Algorithm: {last_ai_algo}", True, (60,60,200))
            screen.blit(txt_algo, (10, status_y))
            status_y += 28
        if winner:
            txt2 = font.render(f"Winner: {winner}", True, RED)
            screen.blit(txt2, (10, status_y))
            status_y += 28
        h3, h4, d3, d4 = threat_analysis(board, AI)
        txt3 = font.render(f"AI open-3s: {h3} 4s: {h4} double-3: {d3} double-4: {d4}", True, RED)
        screen.blit(txt3, (10, status_y))
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()

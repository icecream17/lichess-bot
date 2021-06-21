
from typing import Hashable, List, Union
from strategies import MinimalEngine
import chess
import chess.engine
import time
import math
import random

import logging
logging.root.setLevel(logging.DEBUG)

time_per_turn = (
    1000,
    750,
    555,
    397,
    314,
    238,
    172,
    131,
)

PIECE_VALUES = (
    None,
    1,
    3.5,
    3.7,
    5.7,
    10.2,
    314
)


class Thousandatom(MinimalEngine):
    def __init__(self, *args):
        super().__init__(*args)
        self.nodes = 0
        self.nps = 0


    def search(self, board: chess.Board, timelimit: chess.engine.Limit, _ponder: bool, _draw_offered: bool):
        logging.root.info(f'GOOOOO {board.ply()}')
        starttime = time.time_ns()

        branch = Branch(board)
        result = branch.search(starttime, timelimit)

        self.nodes = branch.nodes
        timetaken = time.time_ns() - starttime
        self.nps = math.inf if timetaken == 0 else self.nodes * 1e9 / timetaken

        if result.move is None:
            logging.root.error(f"WHAT - result of None with \n{board}")
            # return list(board.legal_moves)[0]

        return result

branch_results = {}
class Branch:
    def __init__(self, board: chess.Board, move: Union[chess.Move, None] = None) -> None:
        self.board = board.copy()
        self.hash = to_hashable(board)
        self.legal_moves = list(board.legal_moves)
        self.nodes = 1
        self.move = move
        self.score = evaluate(self.board)
        self.depth = 0
        branch_results[self.hash] = []

    def __str__(self) -> str:
        return f"<Branch ({self.move}) {self.score}>"

    def __repr__(self) -> str:
        return self.__str__()

    def depth1(self):
        self.depth = 1
        branch_results[self.hash].append({})
        for move in self.legal_moves:
            self.nodes += 1
            self.board.push(move)
            branch_results[self.hash][-1][hash(move)] = evaluate(self.board)
            self.board.pop()

    def search(self, starttime: int, timelimit: chess.engine.Limit) -> Union[None, chess.Move]:
        if self.depth == 0:
            self.depth1()

        if len(self.legal_moves) == 1:
            return chess.engine.PlayResult(self.legal_moves[0], None)
        elif len(self.legal_moves) == 0:
            return chess.engine.PlayResult(None, None)

        time_to_take = how_long(timelimit, self.board.turn, len(self.legal_moves))

        print(f"Searching for {time_to_take / 1e6}ms")

        lasttime = time.time_ns() - starttime
        rate = 1
        while lasttime < time_to_take / rate:
            print(self.depth, self.nodes, f"{lasttime / 1e6}ms",
                  self.best_move, self.score)
            if self.step(starttime, time_to_take / rate) == "DONE!":
                break

            oldtime = lasttime
            lasttime = time.time_ns() - starttime
            rate = max(1, lasttime / oldtime)

        print(
            f"    Done! got {self.score} with {self.best_move}\n"
            f"    Depth: {self.depth} ply\n"
            f"    Nodes: {self.nodes}")
        return chess.engine.PlayResult(self.best_move, None)

    def step(self, startime, timeleft):
        if self.depth == 0:
            self.depth1()
        else:
            self.nodes = 0
            self.depth += 1
            self.depthPlus(startime, timeleft)

    def depthPlus(self, starttime, timeleft):
        alpha = -math.inf
        beta = math.inf
        maximizingPlayer = True
        new_results = {}

        self.sort_moves()
        for move in self.legal_moves:
            if starttime - time.time_ns() > timeleft:
                return

            self.nodes += 1
            self.board.push(move)

            result = None
            if self.board.is_game_over(claim_draw=True):
                result = evaluate(self.board)
            else:
                result = -self.nonRootDepthPlus(starttime, timeleft, self.depth - 1, alpha, beta, not maximizingPlayer)

            new_results[hash(move)] = result
            if maximizingPlayer and result > alpha:
                alpha = result
            elif result < beta and not maximizingPlayer:
                beta = result

            self.board.pop()

        branch_results[self.hash].append(new_results)

    def nonRootDepthPlus(self, starttime, timeleft, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or starttime - time.time_ns() > timeleft:
            return evaluate(self.board)

        bestresult = -math.inf
        for move in self.board.legal_moves:
            self.nodes += 1
            self.board.push(move)

            result = None
            if self.board.is_game_over(claim_draw=True):
                result = evaluate(self.board)
            else:
                result = -self.nonRootDepthPlus(starttime, timeleft, depth - 1, alpha, beta, not maximizingPlayer)

            if result > bestresult:
                bestresult = result
            if maximizingPlayer and result > beta:
                self.board.pop()
                return result
            elif result < alpha and not maximizingPlayer:
                self.board.pop()
                return result

            self.board.pop()

        return bestresult

    def sort_moves(self):
        def latest_result(move: chess.Move):
            return branch_results[self.hash][-1][hash(move)]

        # Sort moves in descending score
        self.legal_moves.sort(key=latest_result, reverse=True)

    @property
    def best_move(self):
        self.sort_moves()
        return self.legal_moves[0]


# alpha = best guaranteed by me
# beta = worst guaranteed by opp
evaluations = {}

"""
search
alpha = 7, beta = -7
score = 3
"""
def evaluate(board: chess.Board) -> Union[float, int]:
    position = board._transposition_key()
    if position in evaluations:
        return evaluations[position]

    result = _actual_evaluate(board)
    evaluations[position] = result
    return result

def _actual_evaluate(board: chess.Board, beta=math.inf, depth=1, transpositions: List[Hashable] = []) -> Union[float, int]:
    alpha = -math.inf

    if board.is_checkmate() or board.is_variant_loss():
        return -math.inf
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_variant_draw():
        return 0
    elif board.is_variant_win():
        return math.inf
    elif board.can_claim_draw():
        return 0
    elif depth > 10:
        return static_evaluate(board)

    alpha_has_updated = False
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            position = board._transposition_key()
            if not position in transpositions:
                alpha_has_updated = True
                transpositions.append(position)

                result = None
                if position in evaluations:
                    result = evaluations[position]
                else:
                    result = -_actual_evaluate(board, -alpha, depth + 1, transpositions=transpositions)

                transpositions.pop()
                if result > alpha:
                    alpha = result
                if alpha > beta:
                    board.pop()
                    return alpha  # Opp guaranteed to have score worse than the best
            board.pop()


    if alpha_has_updated:
        return alpha
    else:
        return static_evaluate(board)



def static_evaluate(board: chess.Board):
    if board.is_checkmate() or board.is_variant_loss():
        return -math.inf
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_variant_draw():
        return 0
    elif board.is_variant_win():
        return math.inf
    elif board.can_claim_draw():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color is board.turn:
                score += evaluate_piece(board.turn, piece, square)
            else:
                score -= evaluate_piece(board.turn, piece, square)

    return score


def evaluate_piece(side_to_move: chess.Color, piece: chess.Piece, square: chess.Square) -> Union[float, int]:
    # if piece.piece_type == chess.PAWN:
    #     bonus = 64
    #     if side_to_move is chess.WHITE:
    #         bonus /= (8 - chess.square_rank(square)) ** 2
    #     else:
    #         bonus /= chess.square_rank(square) ** 2

    #     bonus = (bonus / 64) * (PIECE_VALUES[chess.QUEEN] - PIECE_VALUES[chess.PAWN])
    #     return PIECE_VALUES[piece.piece_type] + bonus

    return PIECE_VALUES[piece.piece_type]




def to_ns(timelimit: chess.engine.Limit, side: chess.Color):
    try:
        return timelimit.time * 1e9
    except (AttributeError, TypeError): # TypeError to account for `None`
        if side is chess.WHITE:
            return timelimit.white_clock * 1e9
        else:
            return timelimit.black_clock * 1e9
    except:
        logging.root.error(f"WTF {timelimit}")



# how long should the engine think about a move
def how_long(timelimit: chess.engine.Limit, side: chess.Color, move_count: int):
    ns = to_ns(timelimit, side)
    if len(time_per_turn) > move_count:
        ns /= time_per_turn[move_count]
    else:
        ns /= 100

    ns *= min(move_count, 50) / 10
    return ns


def to_hashable(board: chess.Board):
    return board._transposition_key()


















                                                                                                                                                                                                               # Way too big







############
# Debugging

def main():
    from config import load_config
    from engine_wrapper import create_engine

    config = load_config("./config.yml")

    simulateGame([create_engine(config), create_engine(config)])


def simulateGame(engines):
    import time
    import traceback
    import chess.pgn
    import random

    board = chess.Board()

    for _ in range(7):
        board.push(random.choice(list(board.legal_moves)))

    then = time.time()
    clock = [2 * 60 * 1000] * 2
    inc = [1 * 1000] * 2
    termination = "normal"
    tomove = 0
    draw_offered = [False, False]

    try:
        while not board.is_game_over():
            tomove = tomove ^ 1  # toggle 0 to 1, or 1 to 0

            # timelimit & result are only used once
            timelimit = chess.engine.Limit(time=clock[tomove] / 1000)
            result = engines[tomove].search(board, timelimit, False, draw_offered[tomove ^ 1])
            move = result.move
            draw_offered[tomove] = result.draw_offered

            clock[tomove] -= (time.time() - then) * 1000
            if clock[tomove] <= 0:
                termination = "time forfeit"
                break
            clock[tomove] += inc[tomove]

            then = time.time()

            board.push(move)
            engines[tomove].print_stats()
            print(board)
            print("fen: {}\ntime: {}, inc: {}".format(board.fen(), clock, inc))

        print("Finished!")
        print(board)
        saveGame(board, termination, names=[
                 engine.__class__.__name__ for engine in engines])
    except Exception as error:
        print(traceback.format_exc())
        raise error
    finally:
        print('done')


def saveGame(board, end="normal", names=["BOT your bot" * 2]) -> None:
    from datetime import datetime

    round_no = 0
    with open("logs/count.txt", "r") as gamecount_readfile:
        round_no = int(gamecount_readfile.read())

        with open("logs/count.txt", "w") as gamecount_writefile:
            gamecount_writefile.write(f"{round_no + 1}")

    game_pgn = chess.pgn.Game.from_board(board)
    game_pgn.headers["Event"] = "debug - random opening"
    game_pgn.headers["Site"] = "computer"
    game_pgn.headers["Date"] = datetime.today().strftime('%Y.%m.%d')
    game_pgn.headers["Round"] = round_no
    game_pgn.headers["TimeControl"] = "1800+40"
    game_pgn.headers["Termination"] = end
    game_pgn.headers["White"] = names[0]
    game_pgn.headers["Black"] = names[1]
    print(game_pgn)

    with open("logs/pgns.txt", "a") as pgn_writefile:
        pgn_writefile.write(f"\n\n{game_pgn}")


if __name__ == "__main__":
    print(evaluate(chess.Board(
        "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")))
    main()

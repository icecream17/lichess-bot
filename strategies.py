
import chess
import chess.variant
import random
import time
from engine_wrapper import EngineWrapper


# ms - Not tested
RATE_LIMITING_DELAY = 100


class Prevent427(EngineWrapper):
    def __init__(self, *args):
        super().__init__(*args)
        self.inittime = None
        self.movetimes = []

    def finish(self, output):
        """Call finish when returning to prevent 427 errors"""
        if self.inittime is None:
            self.inittime = time.time()
        else:
            self.movetimes.append(time.time() - self.inittime)

            sleeptime = RATE_LIMITING_DELAY - (
                        self.movetimes[-1] / len(self.movetimes))

            if sleeptime > 0:
                print(f'sleeping for {sleeptime} ms')
                time.sleep(sleeptime / 1000)

        print(f'done! {output}')
        return output


class MinimalEngine(Prevent427):
    def __init__(self, *args):
        super().__init__(*args)
        self.last_move_info = []
        self.engine = self

    def search_with_ponder(self, board, wtime, btime, winc, binc, ponder):
        timeleft = 0
        if board.turn:
            timeleft = wtime
        else:
            timeleft = btime
        return self.search(board, timeleft, ponder)

    # Prevents infinite recursion
    def quit(self):
        pass



# Names from tom7's excellent eloWorld video

class RandomMove(MinimalEngine):
    def search(self, board, *args):
        return self.finish(random.choice(list(board.legal_moves)))


class Alphabetical(MinimalEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return self.finish(moves[0])


# Uci representation is first_move, right?
class FirstMove(MinimalEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return self.finish(moves[0])


class Thousandatom(MinimalEngine):
    def search(self, board, *args):
        print('GOOOOO')
        return self.finish(random.choice(list(board.legal_moves)[::2]))

















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
    clock = [30 * 60 * 1000] * 2
    inc = [40 * 1000] * 2
    termination = "normal"
    tomove = 0

    try:
        while not board.is_game_over():
            tomove = tomove ^ 1  # toggle 0 to 1, or 1 to 0

            move = engines[tomove].search(board, clock[tomove], False)

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
        saveGame(board, termination, names=[engine.__class__.__name__ for engine in engines])
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
    main()

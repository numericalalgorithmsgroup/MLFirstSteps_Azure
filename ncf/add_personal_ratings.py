#!/usr/bin/env python3

import time
import subprocess
import sys


def main():

    scores = {}
    try:
        with open("userscores.txt", 'rt') as fh:
            for line in fh:
                if line.strip():
                    movie, score = line.rsplit('|', maxsplit=1)
                    scores[movie.strip('" ')] =  float(score.strip())
    except OSError as err:
        print("Error failed to open userscores: {}".format(err))
        sys.exit(-1)
    except TypeError as err:
        print("Error: bad score value \"{}\"".format(score))
        sys.exit(-1)

    movies = {}
    with open(sys.argv[1], "rt") as fh:
        for line in fh:
            if line.strip():
                mid, _, name = line.partition(",")
                name, _, tail = name.rpartition(",")
                name = name.strip('" ')
                movies[name] = mid

    for movie in scores:
        if movie not in movies:
            print("Error movie {} not in database.".format(movies))
            sys.exit(-1)

    output = subprocess.check_output(["tail", "-1", sys.argv[2]]).decode()
    output2 = output.split(",")[0]
    uid = int(output2) + 1

    timestamp = int(time.time())
    with open(sys.argv[2], "at") as fh:
        for i, (key, val) in enumerate(scores.items()):
            line = "{},{},{},{}\n".format(uid, movies[key], float(val), timestamp + i)
            print(line, end="")
            fh.write(line)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import sys
import argparse


def do_argparse():

    parser = argparse.ArgumentParser(description="Checker for userscores.txt file")

    parser.add_argument(
        "-m",
        "--moviefile",
        default="./movies.csv",
        type=str,
        help="Path to movies.csv file",
    )
    parser.add_argument(
        "-u",
        "--userscores",
        default="./userscores.txt",
        type=str,
        help="Path to userscores.txt file",
    )

    return parser.parse_args()


def main():

    config = do_argparse()

    print("Checking {} against {}:\n".format(config.userscores, config.moviefile))

    movies = {}
    try:
        with open(config.moviefile, "rt") as fh:
            for line in fh:
                if line.strip():
                    mid, _, name = line.partition(",")
                    name, _, tail = name.rpartition(",")
                    name = name.strip('" ')
                    movies[name] = mid
    except OSError as err:
        if config.moviefile == "./movies.csv":
            print('Failed to open movies.csv (do you need to specify a path with "-m"?)')
        else:
            print("Failed to open {}: {}".format(config.moviefile, err))
        sys.exit(-1)

    numscores = 0
    try:
        with open(config.userscores, "rt") as fh:
            for linenum, line in enumerate(fh, 1):
                if line.strip():
                    numscores += 1  # Can increment here as all errors are fatal
                    try:
                        movie, score = line.rsplit("|")
                    except ValueError:
                        raise ValueError(
                            'bad line format. Need "title|score", (got "{}")'.format(
                                line.strip()
                            )
                        )

                    movie = movie.strip(' "')
                    if not movie in movies:
                        raise ValueError('movie "{}" not in movies.csv'.format(movie))

                    try:
                        score = float(score.strip())
                    except ValueError:
                        raise ValueError(
                            'could not convert "{}" to a number'.format(score.strip())
                        )
                    if score < 0.0 or score > 5.0:
                        raise ValueError(
                            "score must be in range 0-5 (got {})".format(score)
                        )
    except OSError as err:
        if config.userscores == "./userscores.txt":
            print(
                'Failed to open userscores.txt (do you need to specify a path with "-u"?)'
            )
        else:
            print("Failed to open {}: {}".format(config.userscores, err))
        sys.exit(-1)
    except TypeError as err:
        print('Line {}: bad score value "{}"'.format(linenum, score))
        sys.exit(-1)
    except ValueError as err:
        print("Line {}: {}".format(linenum, err))
        sys.exit(-1)

    if numscores < 20:
        print(
            "Warning: Need at least 20 entries for inclusion in model (have {})\n"
            "".format(numscores)
        )

    print("File {}: All entries valid.\n".format(config.userscores))


if __name__ == "__main__":
    main()

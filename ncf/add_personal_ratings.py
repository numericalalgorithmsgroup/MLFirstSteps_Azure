#!/usr/bin/env python3

import time
import subprocess
import sys

scores = {
    "Contact (1997)": 5,
    "Star Trek: First Contact (1996)": 5,
    "Up (2009)": 4,
    "Shrek (2001)": 4,
    "Interstellar (2014)": 4,
    "Jumanji (1995)": 3.5,
    "Star Trek Into Darkness (2013)": 4,
    "Die Hard (1988)": 5,
    "Red (2010)": 5,
    "Wanted (2008)": 4,
    "Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)": 4,
    "Rogue One: A Star Wars Story (2016)": 4,
    "Mad Max: Fury Road (2015)": 5,
    "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)": 4,
    "Thor: Ragnarok (2017)": 5,
    "Guardians of the Galaxy (2014)": 5,
    "Guardians of the Galaxy 2 (2017)": 4.5,
    "Run Lola Run (Lola rennt) (1998)": 4,
    "Edge of Tomorrow (2014)": 3.5,
    "Ghostbusters (a.k.a. Ghost Busters) (1984)": 4,
    "Ghostbusters (2016)": 4.5,
}


def main():

    movies = {}
    with open(sys.argv[1], "rt") as fh:
        for line in fh:
            mid, _, name = line.partition(",")
            name, _, tail = name.rpartition(",")
            name = name.strip('"')
            movies[name] = mid

    timestamp = int(time.time())

    output = subprocess.check_output(["tail", "-1", sys.argv[2]]).decode()
    output2 = output.split(",")[0]
    uid = int(output2) + 1

    with open(sys.argv[2], "at") as fh:
        for i, (key, val) in enumerate(scores.items()):
            line = "{},{},{},{}\n".format(uid, movies[key], float(val), timestamp + i)
            print(line, end="")
            fh.write(line)


if __name__ == "__main__":
    main()

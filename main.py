"""
main.py
-------
Entry point: runs the live hot-fire monitor end-to-end.

    python main.py

This just calls into visualize.py, which internally uses simulator.py to
generate the test data and detector.py to flag anomalies in real time.
"""

from visualize import main

if __name__ == "__main__":
    main()

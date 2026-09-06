"""Answer synthesis over a retrieved evidence set.

baseline: stuff top-k chunks into the prompt, answer.
linkrag:  answer over the linked set, citing each unit's Location, and
          refusing to assert anything not covered by the given evidence.
"""

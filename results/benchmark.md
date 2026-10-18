Means over 20 environments, answering the first k queries on each.
PRM is charged once for building its roadmap.

**Total wall-clock time (ms)**

| planner | k = 1 | k = 5 | k = 10 | k = 20 |
|---|---:|---:|---:|---:|
| PRM | 111 | 134 | 156 | 210 |
| RRT | 194 | 903 | 1849 | 3685 |
| Bi-RRT | 171 | 613 | 1202 | 2669 |

**Total segment collision checks**

| planner | k = 1 | k = 5 | k = 10 | k = 20 |
|---|---:|---:|---:|---:|
| PRM | 18,223 | 18,397 | 18,617 | 19,113 |
| RRT | 1,561 | 7,130 | 14,464 | 29,420 |
| Bi-RRT | 1,670 | 6,154 | 12,030 | 26,724 |

**Success rate and path length over all queries**

| planner | success rate | mean length (solved) |
|---|---:|---:|
| PRM | 95.0% | 6.46 |
| RRT | 96.5% | 8.50 |
| Bi-RRT | 99.0% | 8.71 |

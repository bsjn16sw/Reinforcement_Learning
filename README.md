# rainbow-per-multistep

This repo contains Prioritized Experience Replay(PER) and Multi-step DQN implementations. Those two are variations of DQN that are simply explained in [Rainbow paper](https://arxiv.org/abs/1710.02298). I used base code of [Sung Kim's](https://github.com/hunkim/ReinforcementZeroToAll/blob/master/dqn.py). I also uploaded DDQN code of him for reference.<br>
In sum, here are DDQN, PER, Multi-step DQN implementations. Three of them shares same base code so see and easily compare!

## Double DQN (DDQN)
DDQN codes are from [here](https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py). You can execute by entering `python main_ddqn.py`.

## Prioritized Experience Replay (PER)
Codes related to PER are `per.py`, `sumtree.py`, and `main_per.py`. `per.py` is revision version for PER based on `dqn.py`. `sumtree.py` is from [here](https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/1-dqn/SumTree.py). `main_per.py` is PER implementation based on `main_ddqn.py`. There are several revisions from original. You can execute by entering `python main_per.py`

## Multi-step DQN
For multi-step DQN, I used same DQN class for DDQN: `dqn.py`. `main_nstep.py` is Multi-step DQN implementation based on `main_ddqn.py`. You can execute by entering `python main_nstep.py`.

## References
I refered several links to understand and implement PER and Multi-step DQN. Thanks to:
* https://github.com/hunkim/ReinforcementZeroToAll
* https://github.com/Kyushik/DRL
* https://blog.naver.com/dhrmach45679/221295859279
* https://pemami4911.github.io/paper-summaries/deep-rl/2016/01/26/prioritizing-experience-replay.html
* http://bitly.kr/K6Nz
* http://bitly.kr/9NYX
* https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
* https://wonseokjung.github.io/
* https://sumniya.tistory.com/15?category=781573
* http://www.modulabs.co.kr/?module=file&act=procFileDownload&file_srl=19759&sid=d802d9adf7c768cebac4cc037a010529&module_srl=19712
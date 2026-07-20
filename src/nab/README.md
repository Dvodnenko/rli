### An N-Armed Bandit

#### The problem:
We have a choice of $n$ different options/actions. After each choice we receive a numerical reward,
chosen from a probability distribution. The task is to **maximize the expected reward over some time period**.

#### Some terminology
* *value* - the expected (mean) reward given that some action is selected
* $q(a)$ - the true (actual) value of action $a$
* $Q_t(a)$ - the estimated value of action $a$
* $ε$ - probability of exploring new actions instead of exploiting known ones
* *greedy action* - an action that has the highest value at a given timestep
* *$ε$-greedy method* - a method that selects greedy actions with some probability $ε$
* *greedy method* - a method that **always** selects greedy actions (note that greedy method is an $ε$-greedy method with $ε$=0)

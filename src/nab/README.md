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

#### How it works
One *step* is a cycle: *(1) select & make an action -> (2) observe a reward for that action -> (3) update your beliefs about that action*.

A *run* is a sequence of steps. You can think of it as of "life cycle of your agent".

You start a run and set a number of steps for it, say, 1000 steps. For each run you create **fresh**:
* Array of actual values ($q(a)$'s) of each bandit's arm. If you have $n$ arms, your array must also be of length $n$ - each index stores value of $n$ th arm.
* Array of your estimated values ($Q(a)$'s) for each arm. It must also have length of $n$.
* Array for storing amount of each arm being pulled. Again, with length $n$.
* Array for storing reward after each step. Its length = amount of steps.

After you initialized a run, you make a loop that represents your agent's life cycle; each iteration
makes a step. 

To start a cycle, it should make the step (1) - select & make an action. Since it doesn't
know the real $q(a)$ values, everything it can do - choose random actions and observe their rewards.
The agent (1) makes an action and (2) gets a reward. Then it (3) updates the belief for this action 
and writes it into $Q(a)$. On the first couple of steps the agent just explores all actions and updates
the expected reward (beliefs) for each of them.

As it gets more information, it starts to exploit known actions that give the biggest reward. Typically converges reasonably quickly, though how fast depends on how separated the arm values happen to be. The reason it doesn't get 100% correct answers in the end is that it still explores other actions (with probability ε), even if it knows there is the best one. This ongoing exploration costs some reward in the short term, but protects against permanently misjudging an arm based on noisy early samples.

An *experiment* is a sequence of runs. Its purpose is to find the average reward over lots of runs (in my case, 2000). It stores the reward for each step in each run, and then finds the average reward for $n$ th step in all
runs. The resulting plot's x axis is "# of step", y axis - "take rewards for this $n$ th step from all runs, then average them".

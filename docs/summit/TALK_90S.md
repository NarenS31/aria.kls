# Talk scripts

Plain words. Anything technical is explained the first time it is used. About 155 spoken words a minute: the short version is about 90 seconds, the long one about 3 minutes.

## 90-second version

Most tutoring software talks after every step. We asked: when should an AI math tutor stay quiet?

Our tutor, ARIA, has a math engine that does the algebra itself, so it knows whether each step is right. A right step gets no reply. Only after a wrong step does ARIA think about speaking.

To decide, it uses a very old idea from 1995, called knowledge tracing, which estimates from right and wrong answers how likely a student is to be stuck. We set three cost settings: quiet, balanced and supportive. They say how much we mind interrupting compared with making someone wait. They are settings, not kinds of students.

We tested on old logs of ASSISTments math practice. The grade level is not stated, and it is not calculus. We counted a student as stuck if they did not get three answers right in a row within ten tries. Under that definition, the quiet setting spoke 52% less often than a tutor that helps after every wrong step, and missed 3.7% of the students who did get stuck (missed help is about 4 to 6% across our checks). With another definition of stuck, that saving shrinks a lot.

This is a replay of old data, not a classroom. We have not shown it helps students.

At our table, try "Would you step in?" Eight short moments; nothing is sent anywhere. And if you are a teacher, we would love to talk about a classroom pilot.

## 3-minute version

Most tutoring software talks after every step. We asked the opposite question: when should an AI math tutor stay quiet? Help that comes too early can interrupt a student who was about to work it out. Help that comes too late leaves someone stuck. We wanted to treat that timing as something we can measure.

Our tutor, ARIA, has a math engine that does the algebra itself, so it knows whether each step is right. A right step gets no reply, just a small "watching" sign. Only after a wrong step does ARIA consider speaking, and then it gives one hint that points at the next step. It never gives the student's own answer, and we check every reply for that.

The decision comes from a timing model: Bayesian knowledge tracing, from 1995. From a student's right and wrong answers so far, it estimates how likely they are to still be stuck ten tries later. We did not invent it. Our contribution is the decision rule on top of it, how we tested it, and the results that did not work out. We set the cutoff on one group of students, froze it, and looked at the test group once.

The cutoff depends on a cost setting. Quiet means interrupting is expensive. Balanced means interrupting and waiting cost the same. Supportive means waiting is expensive. They are settings, not types of learners.

We tested on old logs of ASSISTments math practice. The grade level is not stated and it is not calculus, so whether it carries over to our calculus problems is open. We counted a student as stuck if they did not get three answers right in a row within ten tries. Under that definition, on 597 students we had not touched, the quiet setting spoke 52% less often than a tutor that helps after every wrong step, and missed 3.7% of the students who went on to get stuck. Balanced saved 15%, and supportive only 3%.

Now the honest part. With another definition of stuck, such as four right answers in a row, the saving on our practice data falls to about 8%; across the four other rules we tried it ranges from about 7% to 42%. That check is exploratory and used validation data only. The demo also uses one shared set of model settings for every problem, while our test used settings fitted to each skill; on practice data that shared set saves about 44% instead of 50%, so the demo is an approximation of what we tested. Bigger neural network models, including ones that see work on other skills, did not rank who will get stuck any better, and adding how long a student takes to answer did not clear our bar either, so the simple model stays. And a small simulation says the quiet rule only pays off if leaving a stuck student alone is cheap. That is an assumption-dependent check, not evidence about real students.

All of this is replay. We have not shown that any of it helps students learn.

So: try "Would you step in?" at our table. You decide on eight moments, then see what ARIA did; your answers stay in your browser. And if you are a teacher, we would like to talk with one of you about a small classroom pilot, starting with your criticism of our method.

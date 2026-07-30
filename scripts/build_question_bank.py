"""Build ARIA's reviewed, answer-keyed practice bank.

The generated JSON is committed with the product so student practice never
depends on live problem generation. Run this script after editing the task
specifications below.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "question_bank.json"


def task(
    task_id,
    subject,
    topic,
    difficulty,
    problem,
    answer,
    steps,
    ideas,
    mistakes,
    task_type="answer_key",
    coach_hint="",
):
    record = {
        "id": task_id,
        "subject": subject,
        "topic": topic,
        "difficulty": difficulty,
        "task_type": task_type,
        "problem": problem,
        "answer": answer,
        "solution_steps": steps,
        "key_ideas": ideas,
        "common_misconceptions": mistakes,
    }
    if coach_hint:
        record["coach_hint"] = coach_hint
    return record


def linear_equations():
    specs = [
        ("3(x - 4) = 2x + 5", 17, "Distribute 3 to get 3x - 12 = 2x + 5."),
        ("5x + 7 = 2x + 25", 6, "Subtract 2x from both sides to get 3x + 7 = 25."),
        ("4(x + 2) = 3x + 15", 7, "Distribute 4 to get 4x + 8 = 3x + 15."),
        ("7 - 2x = 19", -6, "Subtract 7 from both sides to get -2x = 12."),
        ("(x / 3) + 5 = 11", 18, "Subtract 5 from both sides to get x / 3 = 6."),
        ("6x - 4 = 2(x + 8)", 5, "Distribute 2 to get 6x - 4 = 2x + 16."),
        ("9 - 3(x - 1) = 0", 4, "Distribute -3 to get 9 - 3x + 3 = 0."),
        ("0.5x + 3 = 8", 10, "Subtract 3 from both sides to get 0.5x = 5."),
        ("2(x + 5) - 3 = 15", 4, "Distribute and combine constants to get 2x + 7 = 15."),
        ("4 - (x - 6) = 12", -2, "Distribute the negative sign to get 4 - x + 6 = 12."),
    ]
    result = []
    for index, (equation, answer, first_step) in enumerate(specs, 1):
        result.append(task(
            f"math-algebra-{index:02d}", "Math", "Algebra", "medium",
            f"Solve for x: {equation}", f"x = {answer}",
            [first_step, "Use inverse operations while keeping both sides balanced.",
             f"Check the result by substituting x = {answer} into the original equation."],
            ["equivalent equations", "inverse operations", "checking by substitution"],
            ["Applying an operation to only one side.", "Losing a negative sign while simplifying.",
             "Stopping before the variable is isolated."],
        ))
    return result


def functions_and_systems():
    specs = [
        ("If f(x) = 2x^2 - 3, find f(4).", "29",
         ["Substitute 4 everywhere x appears.", "Compute 2(4^2) - 3 = 32 - 3.", "Simplify to 29."],
         ["Substituting into only one occurrence of x.", "Multiplying 2 by 4 before squaring."]),
        ("A line passes through (2, 3) and (6, 11). Find its slope.", "2",
         ["Use slope = change in y / change in x.", "Compute (11 - 3) / (6 - 2).", "Simplify 8/4."],
         ["Reversing x and y changes.", "Subtracting in different orders."]),
        ("Write the equation of a line with slope 3 through (2, 5).", "y = 3x - 1",
         ["Use y - y1 = m(x - x1).", "Substitute m = 3 and (2, 5).", "Simplify to slope-intercept form."],
         ["Using the point as the y-intercept.", "Changing the slope sign."]),
        ("Solve the system: x + y = 9 and x - y = 3.", "x = 6, y = 3",
         ["Add the equations to eliminate y.", "Solve 2x = 12.", "Substitute x = 6 to find y."],
         ["Adding unlike sides.", "Finding x but not y."]),
        ("Solve the system: 2x + y = 7 and x - y = 2.", "x = 3, y = 1",
         ["Add the equations to eliminate y.", "Solve 3x = 9.", "Substitute x = 3 into either equation."],
         ["Subtracting when addition eliminates y.", "Substituting into the wrong expression."]),
        ("For g(x) = |x - 4|, find g(-2).", "6",
         ["Substitute -2 for x.", "Simplify inside the absolute value: -2 - 4 = -6.", "Take |-6|."],
         ["Treating absolute value as parentheses.", "Dropping the negative before subtracting 4."]),
        ("The table has points (0, 5), (1, 8), and (2, 11). Write its linear rule.", "y = 3x + 5",
         ["Find the constant change in y: +3.", "Use x = 0 to identify the intercept 5.", "Write y = 3x + 5."],
         ["Using 5 as the slope.", "Ignoring that x changes by 1."]),
        ("Find the inverse of f(x) = 2x + 6.", "f^-1(x) = (x - 6) / 2",
         ["Write y = 2x + 6.", "Swap x and y.", "Solve the new equation for y."],
         ["Taking the reciprocal of each term.", "Forgetting to subtract 6 before dividing."]),
        ("What is the vertex of y = (x - 3)^2 + 4?", "(3, 4)",
         ["Recognize vertex form y = (x - h)^2 + k.", "Read h = 3 and k = 4.", "Write the ordered pair."],
         ["Reporting (-3, 4).", "Switching the x- and y-coordinates."]),
        ("An arithmetic sequence begins 7, 11, 15, 19. Write a formula for a_n.", "a_n = 4n + 3",
         ["Find the common difference 4.", "Use a_n = a_1 + (n - 1)d.", "Simplify 7 + 4(n - 1)."],
         ["Using 7 as the common difference.", "Writing 4n + 7 without adjusting for n = 1."]),
    ]
    return [
        task(f"math-functions-{i:02d}", "Math", "Functions and Systems", "medium",
             prompt, answer, steps, ["functions", "representation", "verification"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def geometry_tasks():
    specs = [
        ("A right triangle has legs 6 and 8. Find the hypotenuse.", "10",
         ["Use a^2 + b^2 = c^2.", "Compute 36 + 64 = 100.", "Take the square root of 100."],
         ["Adding the leg lengths.", "Forgetting the square root."]),
        ("A circle has radius 5. Find its area in terms of pi.", "25π square units",
         ["Use A = πr^2.", "Substitute r = 5.", "Square 5 to get 25."],
         ["Using the diameter as the radius.", "Using 2πr, which is circumference."]),
        ("A rectangle is 12 by 7. A 3 by 2 rectangle is cut from one corner. Find the remaining area.", "78 square units",
         ["Find the large area: 12 × 7 = 84.", "Find the cutout area: 3 × 2 = 6.", "Subtract 6 from 84."],
         ["Adding the cutout.", "Subtracting side lengths instead of areas."]),
        ("Two angles form a linear pair. One is 68 degrees. Find the other.", "112 degrees",
         ["Linear-pair angles sum to 180 degrees.", "Compute 180 - 68.", "Check that the pair sums to 180."],
         ["Using 90 degrees.", "Subtracting from 360 degrees."]),
        ("A triangle has angles 45 and 72 degrees. Find the third angle.", "63 degrees",
         ["Triangle angles sum to 180 degrees.", "Add 45 + 72 = 117.", "Compute 180 - 117."],
         ["Subtracting each angle separately from 180.", "Using 360 degrees."]),
        ("A cylinder has radius 3 and height 8. Find its volume in terms of pi.", "72π cubic units",
         ["Use V = πr^2h.", "Substitute r = 3 and h = 8.", "Compute π(9)(8)."],
         ["Using surface area.", "Forgetting to square the radius."]),
        ("Similar triangles have scale factor 3 from small to large. A small side is 5. Find the matching large side.", "15",
         ["Identify the direction of the scale factor.", "Multiply 5 by 3.", "Check that the large side is longer."],
         ["Dividing by 3.", "Adding 3 instead of multiplying."]),
        ("The endpoints of a segment are (-2, 4) and (6, 10). Find the midpoint.", "(2, 7)",
         ["Average the x-coordinates.", "Average the y-coordinates.", "Write the two averages as an ordered pair."],
         ["Finding the distance instead.", "Averaging an x with a y."]),
        ("Find the distance between (1, 2) and (4, 6).", "5",
         ["Find changes: Δx = 3 and Δy = 4.", "Use the distance formula.", "Compute √(3^2 + 4^2) = 5."],
         ["Adding 3 + 4.", "Forgetting the square root."]),
        ("A regular hexagon has side length 4. Find its perimeter.", "24",
         ["A hexagon has six equal sides.", "Multiply 6 by 4.", "Use linear units for perimeter."],
         ["Using an area formula.", "Multiplying by 8 sides."]),
    ]
    return [
        task(f"math-geometry-{i:02d}", "Math", "Geometry", "medium",
             prompt, answer, steps, ["diagram reasoning", "formula selection", "units"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def statistics_tasks():
    specs = [
        ("Find the median of 4, 8, 15, 16, 23, 42.", "15.5",
         ["The values are ordered.", "Average the two middle values, 15 and 16.", "Compute 31 / 2."],
         ["Choosing only one middle value.", "Calculating the mean."]),
        ("Find the mean of 6, 8, 9, 12, 15.", "10",
         ["Add the five values to get 50.", "Divide by the number of values, 5.", "Check that 10 is near the center."],
         ["Dividing by the largest value.", "Forgetting one data point."]),
        ("A bag has 3 red, 5 blue, and 2 green marbles. Find P(red).", "3/10",
         ["Count all marbles: 10.", "Count favorable outcomes: 3.", "Write favorable over total."],
         ["Using 3/7.", "Putting total over favorable."]),
        ("A fair coin is flipped twice. Find the probability of exactly one head.", "1/2",
         ["List HH, HT, TH, TT.", "HT and TH have exactly one head.", "Two of four outcomes gives 1/2."],
         ["Counting HH.", "Assuming there are only two outcomes total."]),
        ("The scores are 70, 72, 74, 76, and 98. Which measure better describes a typical score: mean or median?", "Median",
         ["Notice 98 is much higher than the other scores.", "Recognize that the outlier pulls the mean upward.", "Choose the resistant measure."],
         ["Choosing mean automatically.", "Calling 98 the median."]),
        ("A survey samples 100 students by asking only varsity athletes. What is the main problem?", "Selection bias",
         ["Identify who was eligible to answer.", "Compare that group with the whole student population.", "Name the resulting selection bias."],
         ["Calling the sample random.", "Focusing only on sample size."]),
        ("A spinner has four equal sections labeled A, A, B, C. Find P(A or B).", "3/4",
         ["Count sections labeled A or B.", "There are three favorable sections.", "Divide by four total sections."],
         ["Counting labels instead of sections.", "Multiplying the probabilities."]),
        ("The mean of four numbers is 12. Three numbers are 8, 11, and 15. Find the fourth.", "14",
         ["Use mean × count to find the total: 12 × 4 = 48.", "Add known values: 34.", "Subtract 34 from 48."],
         ["Adding 12 to the known values.", "Dividing the known sum by 4."]),
        ("A scatterplot rises from left to right with points close to a line. Describe the association.", "Strong positive association",
         ["Use direction to identify positive association.", "Use closeness to a line to identify strength.", "Combine strength and direction."],
         ["Calling it negative because y-values vary.", "Claiming causation from association."]),
        ("In a normal distribution, a value has z = -1.5. Is it above or below the mean?", "Below the mean",
         ["Recall that z = 0 is the mean.", "Interpret the negative sign.", "Conclude the value is 1.5 standard deviations below."],
         ["Using the magnitude without the sign.", "Treating z as a raw score."]),
    ]
    return [
        task(f"math-statistics-{i:02d}", "Math", "Statistics and Probability", "medium",
             prompt, answer, steps, ["data interpretation", "reasoning from definitions"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def advanced_math_tasks():
    specs = [
        ("Solve x^2 - 5x + 6 = 0.", "x = 2 or x = 3",
         ["Find two numbers that multiply to 6 and add to -5.", "Factor as (x - 2)(x - 3).", "Set each factor equal to zero."],
         ["Giving only one root.", "Using factors with the wrong signs."]),
        ("Simplify (x^3)(x^5).", "x^8",
         ["The bases are the same.", "Add the exponents 3 + 5.", "Keep the base x."],
         ["Multiplying exponents.", "Changing the base to 2x."]),
        ("Solve 2^x = 32.", "x = 5",
         ["Write 32 as a power of 2.", "Recognize 32 = 2^5.", "Equate the exponents."],
         ["Dividing 32 by 2 once.", "Answering 16."]),
        ("Simplify √72.", "6√2",
         ["Factor 72 as 36 × 2.", "Use √36 = 6.", "Leave √2 in radical form."],
         ["Splitting 72 as 8 × 9 but not finishing.", "Writing 36√2."]),
        ("In a right triangle, opposite = 6 and hypotenuse = 10. Find sin(theta).", "3/5",
         ["Use sin = opposite / hypotenuse.", "Substitute 6/10.", "Reduce the fraction."],
         ["Using adjacent over hypotenuse.", "Flipping the ratio."]),
        ("Convert 150 degrees to radians.", "5π/6",
         ["Multiply degrees by π/180.", "Write 150π/180.", "Reduce by dividing by 30."],
         ["Multiplying by 180/π.", "Dropping π."]),
        ("Find the derivative of f(x) = 3x^2 - 5x + 2.", "f'(x) = 6x - 5",
         ["Apply the power rule to 3x^2.", "Differentiate -5x.", "The derivative of 2 is zero."],
         ["Leaving the constant 2.", "Keeping x^2 unchanged."]),
        ("Evaluate log base 10 of 1000.", "3",
         ["Ask which power of 10 equals 1000.", "Recognize 10^3 = 1000.", "Use the exponent as the logarithm."],
         ["Answering 100.", "Using natural log without reason."]),
        ("Find the sum of the first 6 terms of 2, 5, 8, 11, ...", "57",
         ["Identify a1 = 2, d = 3, and n = 6.", "Find a6 = 17.", "Use S_n = n(a1 + an)/2."],
         ["Using 11 as the sixth term.", "Multiplying the last term by 6."]),
        ("Solve |2x - 1| = 7.", "x = 4 or x = -3",
         ["Create two cases: 2x - 1 = 7 and 2x - 1 = -7.", "Solve each linear equation.", "Check both values."],
         ["Solving only the positive case.", "Writing 2x - 1 = ±1."]),
    ]
    return [
        task(f"math-advanced-{i:02d}", "Math", "Advanced Math", "medium",
             prompt, answer, steps, ["symbolic reasoning", "checking equivalent forms"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def reading_analysis_tasks():
    passages = [
        ("Mara stood outside the crowded cafeteria, rereading the menu though she already knew every item.",
         "How does the detail about rereading the menu reveal Mara's feelings?",
         "It suggests she is delaying entering because she feels anxious or out of place.",
         "She already knows the menu, so rereading it is probably not about food. What might she be delaying?"),
        ("Jon called the broken watch 'perfectly reliable' as it showed noon for the third hour.",
         "What effect does the phrase 'perfectly reliable' create?",
         "It creates verbal irony because the watch is clearly unreliable.",
         "Compare the description 'perfectly reliable' with what the watch actually does. What is the contradiction called?"),
        ("The town planted a single sapling in the empty lot. By spring, neighbors had added a bench, flowers, and a path.",
         "What central idea develops across these sentences?",
         "A small act can inspire a community to improve a shared place.",
         "Track what grows besides the sapling. How does one small action change what the neighbors do?"),
        ("Rain tapped the windows while Priya packed the last photograph into a box and labeled it HOME.",
         "How does the setting contribute to the mood?",
         "The rain and packing create a reflective, melancholy mood connected to leaving home.",
         "Put the rain, the last photograph, and the label HOME together. What feeling do those details create about leaving?"),
        ("The coach never raised her voice. She simply waited, and the noisy gym slowly became still.",
         "What does the coach's response show about her leadership?",
         "Her calm patience gives her authority without needing to shout.",
         "The gym becomes quiet even though she never shouts. What kind of authority does that show?"),
        ("At first, Theo hid his sketches. Later, he pinned one to the classroom wall before anyone arrived.",
         "What change does Theo demonstrate?",
         "He becomes more willing to share his creativity, showing growing confidence.",
         "Compare 'hid' with 'pinned one to the wall.' What personal quality has changed?"),
        ("The article calls the river both a 'highway' and a 'lifeline' for nearby villages.",
         "Why does the author use these two metaphors?",
         "They emphasize that the river provides transportation and resources essential to village life.",
         "Treat the metaphors separately first: what does a highway provide, and what does a lifeline provide?"),
        ("Lena's apology included every detail of what happened except the choice she had made.",
         "What does this sentence imply about Lena?",
         "She is avoiding full responsibility by describing events without owning her decision.",
         "Focus on what Lena leaves out. Why would an apology describe events but omit her own choice?"),
        ("Each morning the factory whistle split the silence; on Sunday, its absence seemed louder.",
         "How does the author use contrast?",
         "The contrast between the usual whistle and Sunday silence emphasizes how deeply the factory shapes daily life.",
         "Why can an absence seem 'louder'? Connect the unusual Sunday silence to the whistle's normal role."),
        ("'Take the map,' Grandfather said, handing Amir a blank sheet of paper.",
         "What can the blank map symbolize?",
         "It can symbolize that Amir must choose or create his own path.",
         "A normal map already shows a route. What does a blank one require Amir to do for himself?"),
    ]
    result = []
    for i, (passage, prompt, target, coach_hint) in enumerate(passages, 1):
        result.append(task(
            f"english-reading-{i:02d}", "English", "Reading Analysis", "medium",
            f'Read: "{passage}"\n\n{prompt}', target,
            ["Identify the exact word, detail, or change named in the question.",
             "Make an inference that fits that evidence.",
             "Explain the connection in one or two precise sentences."],
            ["textual evidence", "inference", "author's craft"],
            ["Repeating the passage without interpreting it.", "Making a claim unsupported by the quoted detail.",
             "Naming a technique without explaining its effect."],
            task_type="writing",
            coach_hint=coach_hint,
        ))
    return result


def argument_writing_tasks():
    specs = [
        ("Revise this thesis to be specific and arguable: Social media affects teenagers in many ways.",
         "A strong revision makes one debatable claim about a specific effect of social media on teenagers.",
         "Replace 'affects' and 'in many ways' with one precise, debatable effect. Which single effect could an essay actually prove?"),
        ("Write a focused claim for an essay arguing whether high schools should start later.",
         "A strong claim takes a clear position and previews a defensible reason related to learning, health, or logistics.",
         "Take a clear yes-or-no position, then attach one reason about learning, health, or logistics. What is your position?"),
        ("Add analysis after this evidence: 'Students who slept eight hours completed the task more accurately.'",
         "A strong analysis explains how the evidence supports a claim connecting sleep with academic performance.",
         "The evidence gives a result, but analysis must explain why it matters. What connection does it suggest between sleep and performance?"),
        ("Revise for precision: School uniforms are good because they help things.",
         "A strong revision replaces 'good' and 'things' with a specific benefit and a clear reason.",
         "Circle the vague words 'good' and 'things.' What exact benefit could replace both of them?"),
        ("Write a counterclaim to: Schools should replace all printed textbooks with tablets.",
         "A strong counterclaim identifies a credible drawback such as access, distraction, cost, or reliability.",
         "A counterclaim needs the strongest reasonable objection, not just the opposite opinion. Which drawback would be hardest to dismiss?"),
        ("Write a rebuttal to the counterclaim that community service requirements take time away from schoolwork.",
         "A strong rebuttal acknowledges the time concern and explains why flexible scheduling or the civic benefit outweighs it.",
         "Acknowledge the time concern first, then answer it. What scheduling choice or larger benefit weakens that objection?"),
        ("Choose the strongest evidence for a claim that trees cool cities: A) trees are green, B) shaded blocks measured lower afternoon temperatures, C) many people like parks.",
         "B is strongest because it directly measures the claimed cooling effect.",
         "Which option directly measures temperature rather than describing trees or people's opinions?"),
        ("Turn this topic into a research question: artificial intelligence in classrooms.",
         "A strong research question narrows the population, use of AI, and outcome that could be investigated.",
         "Narrow three things: which students, which use of AI, and which measurable outcome. What will your study compare?"),
        ("Write a transition between a paragraph about the benefits of public transit and one about its cost.",
         "A strong transition signals contrast while connecting cost to the earlier discussion of benefits.",
         "The relationship is contrast, but the topics must still connect. How can you acknowledge the benefits before introducing cost?"),
        ("Revise this conclusion so it does more than repeat the thesis: Later start times improve student health and learning.",
         "A strong conclusion synthesizes the impact or stakes of later start times without merely repeating the same sentence.",
         "Move from the claim to its larger consequence. What changes for a school community if health and learning both improve?"),
    ]
    return [
        task(
            f"english-argument-{i:02d}", "English", "Argument Writing", "medium",
            prompt, target,
            ["Identify the exact job the sentence must do.", "Draft one focused sentence that does that job.",
             "Check that every word adds a specific idea or logical connection."],
            ["claim", "evidence", "reasoning", "revision"],
            ["Using vague words instead of a defensible idea.", "Adding information that does not serve the requested purpose.",
             "Treating evidence as self-explanatory."],
            task_type="writing",
            coach_hint=coach_hint,
        )
        for i, (prompt, target, coach_hint) in enumerate(specs, 1)
    ]


def language_revision_tasks():
    specs = [
        ("Combine without a comma splice: The experiment ended. The team checked the results.",
         "The experiment ended, and the team checked the results.",
         "Both sides can stand alone as complete sentences. What conjunction or stronger punctuation can join them legally?"),
        ("Fix the misplaced modifier: Running down the hall, the backpack bounced against Maya.",
         "Running down the hall, Maya felt the backpack bounce against her.",
         "Right now the sentence says the backpack is running. Put Maya immediately after the opening phrase so the runner is clear."),
        ("Revise for active voice: The final decision was made by the committee.",
         "The committee made the final decision.",
         "Find the real actor, 'the committee,' and move it before a direct action verb. What did the committee do?"),
        ("Choose the correct word: The new schedule had a positive affect/effect on attendance.",
         "effect",
         "The blank names a result, so the sentence needs a noun. Which option is usually the noun meaning 'result'?"),
        ("Fix subject-verb agreement: The list of proposed changes are on the desk.",
         "The list of proposed changes is on the desk.",
         "Ignore the phrase 'of proposed changes' and find the core subject. Is 'list' singular or plural?"),
        ("Add punctuation: After the storm ended we opened the windows.",
         "After the storm ended, we opened the windows.",
         "The sentence begins with an introductory dependent clause. Where does that clause end?"),
        ("Remove redundancy: The final outcome at the end surprised everyone.",
         "The outcome surprised everyone.",
         "'Outcome,' 'final,' and 'at the end' repeat the same idea. Which one word can carry the meaning by itself?"),
        ("Make the comparison logical: The rainfall in Atlanta is greater than Seattle.",
         "The rainfall in Atlanta is greater than the rainfall in Seattle.",
         "The sentence currently compares rainfall with a city. What matching quantity in Seattle should rainfall be compared with?"),
        ("Fix the unclear pronoun: When Ava called Mia, she was already leaving.",
         "A valid revision names either Ava or Mia as the person who was leaving.",
         "The pronoun 'she' could mean either person. Choose the intended person and replace the pronoun with her name."),
        ("Improve parallel structure: The club values creativity, teamwork, and members who persist.",
         "The club values creativity, teamwork, and persistence.",
         "The first two list items are nouns naming qualities. Turn the third item into the same grammatical shape."),
    ]
    return [
        task(
            f"english-revision-{i:02d}", "English", "Language and Revision", "medium",
            prompt, target,
            ["Identify the sentence-level problem.", "Revise only what is needed.",
             "Read the revision once for clarity and grammar."],
            ["clarity", "sentence structure", "editing"],
            ["Changing the meaning while fixing the sentence.", "Correcting one issue but creating another.",
             "Choosing a word based on sound instead of function."],
            task_type="writing",
            coach_hint=coach_hint,
        )
        for i, (prompt, target, coach_hint) in enumerate(specs, 1)
    ]


def science_tasks():
    specs = [
        ("A car goes from 0 to 20 m/s in 4 seconds. Find its acceleration.", "5 m/s^2",
         ["Use acceleration = change in velocity / time.", "Compute 20 - 0.", "Divide 20 by 4."],
         ["Using final velocity as acceleration.", "Forgetting the time interval."]),
        ("A 2 kg object accelerates at 3 m/s^2. Find the net force.", "6 N",
         ["Use F = ma.", "Substitute 2 kg and 3 m/s^2.", "Multiply and attach newtons."],
         ["Adding mass and acceleration.", "Using kilograms as the final unit."]),
        ("How many moles are in 36 g of water if its molar mass is 18 g/mol?", "2 mol",
         ["Use moles = mass / molar mass.", "Substitute 36 / 18.", "Check units cancel to moles."],
         ["Multiplying by molar mass.", "Using oxygen's mass alone."]),
        ("Balance: __ H2 + __ O2 -> __ H2O.", "2 H2 + 1 O2 -> 2 H2O",
         ["Balance oxygen by placing 2 before H2O.", "Now count four hydrogens on the product side.", "Place 2 before H2."],
         ["Changing subscripts.", "Balancing hydrogen but not rechecking oxygen."]),
        ("Two heterozygous parents Aa × Aa have a child. Find P(aa).", "1/4",
         ["List AA, Aa, Aa, and aa.", "Count one aa outcome.", "Divide by four equally likely outcomes."],
         ["Counting Aa as recessive.", "Using 1/2 without a Punnett square."]),
        ("A plant cell is placed in a hypotonic solution. What happens to water movement?", "Water moves into the cell",
         ["Compare solute concentration outside and inside.", "Use osmosis: water moves toward higher solute concentration.", "Conclude net water movement is inward."],
         ["Saying solute moves by osmosis.", "Reversing hypotonic and hypertonic."]),
        ("Why do seasons occur on Earth?", "Earth's axial tilt changes the angle and duration of sunlight during its orbit",
         ["Identify Earth's axial tilt.", "Connect tilt to sunlight angle and day length.", "Distinguish this from Earth-Sun distance."],
         ["Saying Earth is much closer to the Sun in summer.", "Explaining day and night instead."]),
        ("A population exceeds its carrying capacity. Predict the likely short-term effect.", "Resource competition rises and population growth slows or declines",
         ["Recall that carrying capacity reflects available resources.", "Recognize that excess population increases competition.", "Predict lower survival or reproduction."],
         ["Assuming unlimited exponential growth.", "Treating carrying capacity as a minimum."]),
        ("In an experiment, fertilizer amount changes and plant height is measured. Name the independent variable.", "Amount of fertilizer",
         ["Identify what the researcher changes.", "Separate it from the measured outcome.", "Name fertilizer amount."],
         ["Choosing plant height.", "Naming an uncontrolled variable."]),
        ("A wave has frequency 5 Hz and wavelength 2 m. Find its speed.", "10 m/s",
         ["Use wave speed = frequency × wavelength.", "Substitute 5 × 2.", "Attach meters per second."],
         ["Dividing wavelength by frequency.", "Using hertz as the final unit."]),
    ]
    return [
        task(f"science-{i:02d}", "Science and Coding", "Science Reasoning", "medium",
             prompt, answer, steps, ["model selection", "evidence", "units"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def coding_tasks():
    specs = [
        ("What does Python print? nums = [2, 4, 6]; print(nums[1])", "4",
         ["Python list indexes start at 0.", "Index 1 is the second item.", "The second item is 4."],
         ["Choosing 2 because it is first.", "Choosing 6 by counting from 1."]),
        ("What does Python print? total = 0\nfor n in [1, 2, 3]:\n    total += n\nprint(total)", "6",
         ["Start total at 0.", "Add 1, then 2, then 3.", "Print the final total."],
         ["Reporting the last item only.", "Forgetting the loop updates total."]),
        ("Fix the boundary bug: for i in range(len(items) + 1): print(items[i])", "Use range(len(items))",
         ["Valid indexes stop at len(items) - 1.", "The current range includes len(items).", "Remove the + 1."],
         ["Changing +1 to -1 and skipping the last item.", "Blaming the list itself."]),
        ("What does this condition test? if age >= 13 and age <= 19:", "Whether age is between 13 and 19 inclusive",
         ["Read each comparison separately.", "The and operator requires both to be true.", "Include both endpoints."],
         ["Treating and as or.", "Excluding 13 and 19."]),
        ("A function should return the square of x. Complete: def square(x): return ____", "x * x",
         ["Use the parameter x.", "Multiply x by itself.", "Return the expression."],
         ["Printing instead of returning.", "Using x + x."]),
        ("What is the time complexity of binary search on a sorted list?", "O(log n)",
         ["Binary search halves the remaining search space.", "Count how often n can be halved.", "Recognize logarithmic growth."],
         ["Answering O(n) because the list has n items.", "Ignoring the sorted-list requirement."]),
        ("Why does this loop never end? while x > 0: print(x)", "x never changes inside the loop",
         ["Identify the loop condition.", "Check whether x is updated.", "Notice the condition can remain true forever."],
         ["Blaming print.", "Assuming Python automatically decrements x."]),
        ("Which structure is best for mapping student IDs to names: list, dictionary, or stack?", "Dictionary",
         ["The task uses unique keys and associated values.", "Dictionaries support direct key lookup.", "Map each ID key to a name value."],
         ["Choosing a stack because it stores items.", "Using list position as if it were an ID."]),
        ("What does SQL `SELECT name FROM students WHERE grade >= 90;` return?", "Names of students whose grade is at least 90",
         ["SELECT identifies the returned column.", "FROM identifies the table.", "WHERE filters rows using an inclusive comparison."],
         ["Returning every student column.", "Excluding students with exactly 90."]),
        ("A test expects add(2, 3) == 5 but receives 6. What should you inspect first?", "The implementation of add and how it combines its two inputs",
         ["Use the failing example to locate the smallest relevant function.", "Compare expected addition with the returned value.", "Inspect the operation and any extra offset."],
         ["Changing the test to expect 6.", "Debugging unrelated code first."]),
    ]
    return [
        task(f"coding-{i:02d}", "Science and Coding", "Coding Reasoning", "medium",
             prompt, answer, steps, ["trace execution", "debugging", "precise language"], mistakes)
        for i, (prompt, answer, steps, mistakes) in enumerate(specs, 1)
    ]


def build():
    tasks = (
        linear_equations()
        + functions_and_systems()
        + geometry_tasks()
        + statistics_tasks()
        + advanced_math_tasks()
        + reading_analysis_tasks()
        + argument_writing_tasks()
        + language_revision_tasks()
        + science_tasks()
        + coding_tasks()
    )
    assert len(tasks) == 100, f"Expected 100 tasks, found {len(tasks)}"
    ids = [item["id"] for item in tasks]
    assert len(ids) == len(set(ids)), "Task IDs must be unique"
    assert all(len(item["solution_steps"]) >= 3 for item in tasks)
    assert all(len(item["common_misconceptions"]) >= 2 for item in tasks)
    OUTPUT.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tasks)} tasks to {OUTPUT}")


if __name__ == "__main__":
    build()

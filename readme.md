# Similarity-between-Bug-Description-and-Methods

Motivation
- Each developer is faced with the need to fix bugs in developed programs.
- Developers spend quite a lot of their work time searching for the location and fixing bugs. (McConnell, S. (1993). Code Complete)
- Even one hundred percent coverage of the code with tests does not guarantee the absence of errors.
- Reducing the time spent searching for functions containing bugs (faulty components) will save time for solving more complex problems. In addition, it will significantly reduce the cost of software development.

Goal
- The most popular  version control version control system is GIT. The commit description (commit message) indicates which changes have been made and which functions.
- When a new bug is detected in the system, a description of the bug is attached. This description is literally written. All bugs can be found in the bug management system, such as JIRA.
- Diagnosis of bugs in a code that based on a similarity of a text in which a bugs are listed and a text of a commites containing the components.
- Hypothesis: Description of bugs and commit messages will allow to identify functions with bugs (faulty components).

Solution Method
- Our solution path is based on topic modeling methods.
- This is a good artificial intelligence tool for automatic text analysis, which allows to solve many problems.
- We use it to solve the problem of diagnosis bugs in code.
- It is with the help of topic modeling that we implement a recommendation system that will predict faulty system components - functions with bugs.




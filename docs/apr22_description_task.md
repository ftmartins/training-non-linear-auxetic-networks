# Tasks

## Histogram normalized s_tot for gloabl vs local tasks

- Make a histogram plot of normalized |S_tot| (log-spaced bins); different colors for the global (my tasks 0-9) and local tasks (ref data in ToFelipe0421)
- Make a histogram of normalized |psi_i| (log-spaced bins).

## Notebook to train allosteric tasks like Stephen

- Consider the scripts in ToFelipe0422.
- Make a notebook where a network will be created like in EnsembleGenerator030726.ipynb
- Then, in the same notebook, do gradient descent to optimize the spring stiffnesses (do not use lammps, use our elasticNetwork module and other such already existing code to setup the network). The cost function to be optimized is defined by 'mse = (np.linalg.norm(nodesfree[2]-nodesfree[3])-tod)**2' 
- This notebook should optimize stiffnesses with gradient descent on the cost. Make three realizations, 3 different stiffness initial condition.
- At the end, analyze the realizations as in tofelipe0421_analysis.ipynb; for which you will need to calculate the total susceptibility for each edge at the subtask compression strains.
- Be careful, the actuation protocol is different that for my old auxetic tasks; one is straining a single edge away from its rest length.


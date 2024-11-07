p_w1 = int(input("Enter a priori percentage of spam emails: ")) / 100
p_w2 = int(input("Enter a priori percentage of non-spam emails: ")) / 100

print("p_w1 = ", p_w1)
print("p_w2 = ", p_w2)

# Define actions
a1 = "classify as spam"
a2 = "classify as non-spam"

print("\na1 = ", a1)
print("a2 = ", a2)
print("")

# Define a default loss matrix
lossMatrix_11 = 0 
lossMatrix_12 = 1 
lossMatrix_21 = 5 
lossMatrix_22 = 0 

print("Loss Matrix:")
print(lossMatrix_11, lossMatrix_12)
print(lossMatrix_21, lossMatrix_22)
print("")

def getBayesianRiskOptimalAction():
    R_a1 = (lossMatrix_11 * p_w1) + (lossMatrix_12 * p_w2)
    R_a2 = (lossMatrix_21 * p_w1) + (lossMatrix_22 * p_w2)
    print("R_a1 = ", R_a1)
    print("R_a2 = ", R_a2)
    print("")

    if R_a1 < R_a2:
        return a1
    else:
        return a2

def getConditionalRiskOptimalAction():
    # Define default conditional probabilities
    p_x1_by_w1 = 0.8 
    p_x1_by_w2 = 0.3 
    p_x2_by_w1 = 0.2 
    p_x2_by_w2 = 0.7 

    # for x1
    R_a1_by_x1 = lossMatrix_11 * (p_x1_by_w1 * p_w1) + lossMatrix_12 * (p_x1_by_w2 * p_w2)
    R_a2_by_x1 = lossMatrix_21 * (p_x1_by_w1 * p_w1) + lossMatrix_22 * (p_x1_by_w2 * p_w2)

    print("R_a1_by_x1 = ", R_a1_by_x1)
    print("R_a2_by_x1 = ", R_a2_by_x1)

    x1_optimal_action = a1 if R_a1_by_x1 < R_a2_by_x1 else a2
    print("For x1 (email with spam-indicative word), optimal action = ", x1_optimal_action)
    print("")

    # for x2
    R_a1_by_x2 = lossMatrix_11 * (p_x2_by_w1 * p_w1) + lossMatrix_12 * (p_x2_by_w2 * p_w2)
    R_a2_by_x2 = lossMatrix_21 * (p_x2_by_w1 * p_w1) + lossMatrix_22 * (p_x2_by_w2 * p_w2)

    print("R_a1_by_x2 = ", R_a1_by_x2)
    print("R_a2_by_x2 = ", R_a2_by_x2)

    x2_optimal_action = a1 if R_a1_by_x2 < R_a2_by_x2 else a2
    print("For x2 (email without spam-indicative word), optimal action = ", x2_optimal_action)

print("Bayesian Risk")
print(getBayesianRiskOptimalAction(), " is the optimal action")

print("\nConditional Risk")
getConditionalRiskOptimalAction()

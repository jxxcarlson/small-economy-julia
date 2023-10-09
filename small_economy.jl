#=
small_economy:
- Julia version: 
- Author: carlson
- Date: 2023-10-08
=#


struct Model
    xMax::Float64
    yMax::Float64
    initialBalance::Float64
    numberOfAgents::Int64
    Δm :: Float64
    transactionsToRun::Int64
    numberOfQuantiles::Int64
    makeTransaction::Function
    report::Function
end


# Named-argument constructor with default values
function Model(;xMax=100.0, yMax=100.0, initialBalance=10.0, numberOfAgents=100, Δm=1.0, transactionsToRun=1000,
    numberOfQuantiles = 5, makeTransaction=makeSimpleTransaction, report=standardReport)
    return Model(xMax, yMax, initialBalance, numberOfAgents, Δm, transactionsToRun, numberOfQuantiles, makeTransaction, report)
end



# Define the struct for your "thing"
mutable struct Agent
    i:: Int64
    x::Float64
    y::Float64
    balance::Float64
end


function distinctRandomPair()
    (i,j) = rand(1:model.numberOfAgents, 2)
    while i == j
      (i,j,)= rand(1:model.numberOfAgents, 2)
    end
    return (i,j)
end

function makeSimpleTransaction(state)
    i, j = distinctRandomPair()
    agentA = state[i]
    agentB = state[j]
    if agentA.balance - model.Δm >= 0
        agentA.balance -= model.Δm
        agentB.balance += model.Δm
    end
    return state
end

function runTransactions(n::Int64, state::Vector{Agent})::Vector{Agent}
    for i in 1:n
        model.makeTransaction(state)
    end
    return(state)
end

function getBalances(agents::Vector{Agent})::Vector{Float64}
    return map(agent -> agent.balance, agents)
end


init = function (model::Model) # ::Vector{Agent}
    # Create an array of "Agent" objects
    state = Vector{Agent}(undef, model.numberOfAgents)  # Initialize an empty array of `Thing`
    # Populate the array
    for i in 1:model.numberOfAgents
        state[i] = Agent(i, model.xMax*rand(), model.yMax*rand(),  model.initialBalance)
    end
    return state
end

runN_ = function(model::Model)  # ::Vector{Agent}
  state = init(model)
  return runTransactions(model.transactionsToRun, state)
end

printBalances = function(state)
    balances = sort(getBalances(state::Vector{Agent}))
    for i in 1:model.numberOfAgents
        println(i, ": ", balances[i])
    end
end

# quantiles = function(numberOfQuantiles::Int32, data::Vector{Float})::Vector{Int}
function quantiles(numberOfQuantiles::Int, data::Vector{Float64})
    sorted_data = sort(data)
    output = Vector{Vector{Float64}}(undef, numberOfQuantiles)
    n = length(sorted_data)
    Δn = Int(floor(n / numberOfQuantiles))

    for i = 1:(numberOfQuantiles - 1)
        first = (i - 1) * Δn + 1
        last = i * Δn
        output[i] = sorted_data[first:last]
    end

    # Handle the last quantile
    output[numberOfQuantiles] = sorted_data[((numberOfQuantiles - 1) * Δn + 1):end]

    return output
end
    
function average(data::Vector{Float64})::Float64
    n = length(data)
    return (sum(data)/n)
end


function normalizeData(data::Vector{Float64})::Vector{Float64}
    mean = average(data)
    normalizedData = (x -> x - mean).(data)
    return(normalizedData)
end

function variance(data::Vector{Float64})::Float64
    normalized = normalizeData(data)
    squares = (x -> x^2).(normalized)
    numerator = sum(squares)
    denominator = length(squares)
    return(numerator/denominator)
end

function stdev(data::Vector{Float64})::Float64
    return(sqrt(variance(data)))
end

function quantileAverages(numberOfQuantiles::Int, data::Vector{Float64})
    return(average.(quantiles(numberOfQuantiles, data)))
end




function printModel(model)
    println("Initial per capita balance   : ", model.initialBalance)
    println("Number of agents:            : ", model.numberOfAgents)
    println("Inital per capita balance    : ", model.initialBalance)
    println("Amount of transaction        : ", model.Δm)
    println("transactionsToRun (millions) : ", model.transactionsToRun / 1_000_000)

end

function successiveRatios(data::Vector{Float64})
    n = length(data)
    ratios = Vector{Float64}(undef, n - 1)
    for i = 1:(n - 1)
        # ratios[i] = data[i+1] / data[i]
        ratios[i] = data[i] / data[i+1]
    end
    return ratios
end

using LinearAlgebra  # for the `norm` function

function wealth_entropy(quantiles::Vector{Float64})
    total_wealth = sum(quantiles)
    
    # Calculate "probabilities" (shares of total wealth)
    p = quantiles ./ total_wealth
    
    # Calculate entropy
    entropy = -sum(p .* log2.(p .+ eps()))  # eps() is added to handle log(0)
    
    return entropy
end

function gini_index(data::Vector{Float64})
    n = length(data)
    mean_value = average(data)
    
    sum_diff = 0.0
    for i in 1:n
        for j in 1:n
            sum_diff += abs(data[i] - data[j])
        end
    end
    
    gini = sum_diff / (2 * n^2 * mean_value)
    return gini
end


function standardReport(state)
    numberOfAngentsPerQuantile = model.numberOfAgents / model.numberOfQuantiles
    totalWealth = model.initialBalance * model.numberOfAgents
    


    println("\n")
    printModel(model)

    modelQuantiles = quantiles(model.numberOfQuantiles, getBalances(state))
    averageByQuantile = quantileAverages(model.numberOfQuantiles, getBalances(state))
    fractionalWealthByQuantile = (x -> x/Float64(totalWealth)).(averageByQuantile*numberOfAngentsPerQuantile)


    println("\nTotal wealth: ", totalWealth)
    println("\nnumberOfAngentsPerQuantile: ", numberOfAngentsPerQuantile)
    println("\nMoney by percentile:")

    for i = 1:model.numberOfQuantiles
        println(model.numberOfQuantiles + 1 - i, ": ", 
          round(averageByQuantile[model.numberOfQuantiles + 1 - i], digits = 2), ", ",
          round(fractionalWealthByQuantile[model.numberOfQuantiles + 1 - i], digits = 2))
    end

    println("\nSum of money by percentile: " ,round(sum(averageByQuantile), digits =0), "\n")

    gini = gini_index(averageByQuantile)
    println("\nGini index: ", round(gini, digits = 2))
    entropy = wealth_entropy(averageByQuantile)
    println("Entropy: ", round(entropy, digits = 2))

 
    println("\nExamine fit with Boltzmann-Gibbs distribution:")
    ratios = successiveRatios(reverse(averageByQuantile))
    println("Inter percentile ratios:")
    for i = 1:model.numberOfQuantiles - 1 
        println(i, ": ", round(ratios[i], digits=2))
    end
    
    println("\n")

    println("Mean of ratios: ", round(average(ratios), digits = 2))
    println("Standard deviation of ratios: ", round(stdev(ratios), digits = 2))

    println("\n")
 
end

function runN(model)
    state = runN_(model)
    model.report(state)
end


############ ############ ############ ############ ############ 
#                 INPUT, COMPUTATION, AND OUTPUT
############ ############ ############ ############ ############ 

# # struct Model
# #     xMax::Float64
# #     yMax::Float64
# #     initialBalance::Float64
# #     numberOfAgents::Int64
# #     Δm :: Float64
# #     transactionsToRun::Int64
# #     makeTransaction::Function
# # end


model = Model( numberOfAgents = 10_000, Δm = 1.0, initialBalance = 100, transactionsToRun = 1_000_000, 
   numberOfQuantiles = 100, makeTransaction = makeSimpleTransaction)

runN(model)





#=
small_economy:
- Julia version: 
- Author: carlson
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
    socialSecurity::Function
    socialSecurityTaxRate::Float64
    taxationInterval::Int64
    report::Function
    reportInterval::Int64
end


# Named-argument constructor with default values
function Model(;xMax=100.0
               , yMax=100.0
               , initialBalance=10.0
               , numberOfAgents=100
               , Δm=1.0
               , transactionsToRun=1000
               , numberOfQuantiles=5
               , makeTransaction=makeSimpleTransaction
               , socialSecurity=standardSocialSecurity
               , socialSecurityTaxRate=0.0
               , taxationInterval=1000
               , report=briefReport
               , reportInterval=10)
    return Model(
          xMax
        , yMax
        , initialBalance
        , numberOfAgents
        , Δm
        , transactionsToRun
        , numberOfQuantiles
        , makeTransaction
        , socialSecurity
        , socialSecurityTaxRate
        , taxationInterval
        , report
        , reportInterval)


end




# Define the struct for your "thing"
mutable struct Agent
i:: Int64
x::Float64
y::Float64
balance::Float64
end

mutable struct State 
    step::Int64
    agents::Vector{Agent}
end

function nextState(model::Model, state::State)::State
    new_step = state.step + 1
    new_agents = model.makeTransaction(state.agents)
    if model.socialSecurityTaxRate > 0 && new_step % model.taxationInterval == 0
        new_agents2 = model.socialSecurity(model, new_agents)
    else
        new_agents2 = new_agents
    end
    return State(new_step, new_agents2)
end

function standardSocialSecurity(model::Model, agents::Vector{Agent})
    revenue = sum((agent -> model.socialSecurityTaxRate * agent.balance).(agents))
    payment = revenue/model.numberOfAgents
    new_agents = (agent -> updateAgent!(agent, payment, model.socialSecurityTaxRate)).(agents)
    return(new_agents)
end

function updateAgent!(agent::Agent, payment::Float64, taxRate::Float64)::Agent
    newBalance = agent.balance - agent.balance * taxRate + payment
    newAgent = update_balance!(agent, newBalance)
    return(newAgent)
end

function update_balance!(agent::Agent, new_balance::Float64)
    agent.balance = new_balance
    return(agent)
end

function makeSimpleTransaction(agents)
    i, j = distinctRandomPair()
    agentA = agents[i]
    agentB = agents[j]
    if agentA.balance - model.Δm >= 0
        agentA.balance -= model.Δm
        agentB.balance += model.Δm
    end
    return agents
end

function run(model::Model, n::Int64)::State
    state = initialState(model) 
    while state.step < n
        state = nextState(model, state)
        if state.step %  model.reportInterval  == 0  
           model.report(state, model)
        end

    end
    return(state)
end

function initialState(model::Model)::State
    agents = Vector{Agent}(undef, model.numberOfAgents)
    for i in 1:model.numberOfAgents
        agents[i] = Agent(i, model.xMax*rand(), model.yMax*rand(), model.initialBalance)
    end
    return State(0, agents)
end

######################################################


function ubi(state::State, model::Model)::State
   newState = state
   return(newState)
end
######################################################

function distinctRandomPair()
    (i,j) = rand(1:model.numberOfAgents, 2)
    while i == j
      (i,j,)= rand(1:model.numberOfAgents, 2)
    end
    return (i,j)
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


function standardReport(state::State, model)
    numberOfAngentsPerQuantile = model.numberOfAgents / model.numberOfQuantiles
    totalWealth = model.initialBalance * model.numberOfAgents
    


    println("\n")
    printModel(model)

    modelQuantiles = quantiles(model.numberOfQuantiles, getBalances(state.agents))
    averageByQuantile = quantileAverages(model.numberOfQuantiles, getBalances(state.agents))
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

function briefReport(state, model)
    numberOfAngentsPerQuantile = model.numberOfAgents / model.numberOfQuantiles
    totalWealth = model.initialBalance * model.numberOfAgents
    


    println("\n")
    printModel(model)

    modelQuantiles = quantiles(model.numberOfQuantiles, getBalances(state.agents))
    averageByQuantile = quantileAverages(model.numberOfQuantiles, getBalances(state.agents))
    fractionalWealthByQuantile = (x -> x/Float64(totalWealth)).(averageByQuantile*numberOfAngentsPerQuantile)

  
    gini = gini_index(averageByQuantile)
    entropy = wealth_entropy(averageByQuantile)
    println("\nGini index: ", round(gini, digits = 2), "  Entropy: ", round(entropy, digits = 2))

 

    println("\nFit with Boltzmann-Gibbs distribution:")
    ratios = successiveRatios(reverse(averageByQuantile))
    println("Mean of ratios: ", round(average(ratios), digits = 2))
    println("Standard deviation of ratios: ", round(stdev(ratios), digits = 2))

    println("\nAdjusted fit with Boltzmann-Gibbs distribution:")
    filteredAverages = filter(x -> x >= 0.5, averageByQuantile)
    println("Quantiles dropped: ", length(averageByQuantile) - length(filteredAverages))
    filteredRatios = successiveRatios(reverse(filteredAverages))
    println("Mean of ratios: ", round(average(filteredRatios), digits = 2))
    println("Standard deviation of ratios: ", round(stdev(filteredRatios), digits = 2))


    println("\n")
 
end

function onelineReport(state::State, model::Model)
    ## global model  # Declare model if it is a global variable
    
    numberOfAgentsPerQuantile = model.numberOfAgents / model.numberOfQuantiles
    totalWealth = model.initialBalance * model.numberOfAgents

    modelQuantiles = quantiles(model.numberOfQuantiles, getBalances(state.agents))
    averageByQuantile = quantileAverages(model.numberOfQuantiles, getBalances(state.agents))
    fractionalWealthByQuantile = (x -> x / Float64(totalWealth)).(averageByQuantile * numberOfAgentsPerQuantile)

    ratios = successiveRatios(reverse(averageByQuantile))

    gini = gini_index(averageByQuantile)
    entropy = wealth_entropy(averageByQuantile)

    if state.step % model.reportInterval == 0
            println(
            state.step,
            ": gini: ", 
            round(gini, digits = 2), 
            ", entropy: ", 
            round(entropy, digits = 2),
            ", hi/lo: ", round(hilo(identity(averageByQuantile))))
            # ", quintiles: ", (x -> round(x, digits=2)).(reverse(averageByQuantile)))
    end
end

function hilo(list::Vector{Float64})
    n = length(list)
    if n < 2  # To avoid division by zero or index out-of-bounds
        return 0.0
    end
    ratio = list[n] / list[1]
    # println(list[n], ", ", list[1])
    return round(ratio, digits=4)
end

function report_state(state, model)
    if state.step % model.reportInterval == 0
        gini = compute_gini(state)  # Assume you have a function to compute this
        entropy = compute_entropy(state)  # Assume you have a function to compute this

        println(
            state.step, 
            ": gini: ", 
            round(gini, digits = 2), 
            ", entropy: ", 
            round(entropy, digits = 2)
        )
    end
end


function runN(model)
    state = runN_(model)
    model.report(state, model)
end


############ ############ ############ ############ ############ 
#                 INPUT, COMPUTATION, AND OUTPUT
############ ############ ############ ############ ############ 

model = Model(
    numberOfAgents = 200
    , Δm = 1.0
    , initialBalance = 10
    , transactionsToRun = 1_000_000
    , numberOfQuantiles = 5
    , makeTransaction = makeSimpleTransaction
    , socialSecurityTaxRate = 0.01
    , report = onelineReport #  report_state # briefReport# onelineReport
    , reportInterval = 1000
    )


runN(model)

# run(model, 100_000)

#=
small_economy:
- Julia version: 
- Author: carlson
=#

using Plots

######################################################

### TYPES: Model, Agent, State

#=
The State data structures is for a "state machine" with 
function

    nextState(model::Model, state::State)::State

=#

######################################################

mutable struct Model
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
    showEndReport::Bool
    showPlot::Bool
    verbose::Bool
    printModel::Bool
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
               , reportInterval=10
               , showEndReport=false
               , showPlot=false
               , verbose=false
               , printModel=false)
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
        , reportInterval
        , showEndReport
        , showPlot
        , verbose
        , printModel)


end


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

### THE NEXT STATE FUNCTION AND ITS HELPERS

function nextState(model::Model, state::State)::State
    new_step = state.step + 1
    new_agents = model.makeTransaction(model, state.agents)
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

function makeSimpleTransaction(model, agents)
    i, j = distinctRandomPair(model)
    agentA = agents[i]
    agentB = agents[j]
    if agentA.balance - model.Δm >= 0
        agentA.balance -= model.Δm
        agentB.balance += model.Δm
    end
    return agents
end

######################################################

### RUN THE MACHINE N TIMES

function run(settings::String)
    model = if settings == ""
                Model()
            else
                setModel(settings)
            end
    if model.printModel
        printModel(model)
    end
    state = initialState(model) 

    while state.step < model.transactionsToRun
        
        if model.verbose && state.step % model.reportInterval  == 0  
            model.report(state, model)
        end
        if state.step % 10_000_000 == 0
            print("\n", Int(1 + state.step / 10_000_000), " ")
        end
        if state.step % 100_000 == 0
            print(".")
        end
        if state.step % 1_000_000 == 0
            print(" * ")
        end

        state = nextState(model, state)
        
    end
    print("\n")

    if model.showEndReport
        standardReport(state, model)
    end
    if model.showPlot 
        plotDistribution(state, model)
        println("\nplot saved to current directory\n")
    end
    # return(state)
end

function initialState(model::Model)::State
    agents = Vector{Agent}(undef, model.numberOfAgents)
    for i in 1:model.numberOfAgents
        agents[i] = Agent(i, model.xMax*rand(), model.yMax*rand(), model.initialBalance)
    end
    return State(0, agents)
end

function distinctRandomPair(model)
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


######################################################

### REPORT HELPERS

######################################################

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
    println("\nInitial per capita balance   : ", model.initialBalance)
    println("Number of agents:            : ", model.numberOfAgents)
    println("Inital per capita balance    : ", model.initialBalance)
    println("Amount of transaction        : ", model.Δm)
    println("TransactionsToRun (millions) : ", transactionsToRunString(model))
    println("Quantiles                    : ", model.numberOfQuantiles)


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


##### EXPONENTIAL DISTRIBUTION ESTIMATORS #####

function differences(data::Vector{Float64})::Vector{Float64}
    n = length(data) - 1
    output = Vector{Float64}(undef, n)
    for i in 1:n
        output[i] = data[i+1] - data[i]
    end
    return(output)
end

function k_estimator(data::Vector{Float64})::Float64
    log_data = log.(data)
    diffs = differences(log_data)
    return(average(diffs))
end

function k_estimator2(data::Vector{Float64})::Float64
    return(1/average(data))
end

function k_estimator3(data::Vector{Float64})::Float64
    n = length(data)
    return((n - 1)/(n * average(data)))
end

######################################################

##### PLOT #####

######################################################

function plotDistribution(state, model)
    averageByQuantile = quantileAverages(model.numberOfQuantiles, getBalances(state.agents))
    reversed = reverse(averageByQuantile)
    A = reversed[1]
    k = k_estimator3(averageByQuantile)
    # k = k_^1.2
    # println("k_: ", k_, ", k: ", k)
    println("A: ", round(A, digits = 3))
    println("k: ",round(k, digits = 3))

    function expo(x)
        return A*exp(-k*(x - 1))
    end
    xs = 1:length(reversed)
    us = reversed
    vs = expo.(xs)
    money = "m:" * string(model.initialBalance) * ", "
    transaction = "tr:" * string(model.Δm) * ", "
    tax = "tx:" * string(model.socialSecurityTaxRate)  * ", "
    population = "p:" * string(model.numberOfAgents) * ", "
    n_agents = "n:" * transactionsToRunString(model)
    gini_ = round(gini_index(quantileAverages(model.numberOfQuantiles, getBalances(state.agents))), digits = 2) 
    gini = "gini:" * string(gini_) * ", "


    titleString = "\n" * money * transaction * tax * population * gini * n_agents
    filename_ = replace(titleString, "\n" => "")
    filename = replace(filename_, ":" => "")
    
	plot(xs, [us, vs], titlefont=("Courier", 8), legend=:topright, title=titleString)
    savefig(filename * ".png")
	
end

function transactionsToRunString(model::Model)::String
    if model.transactionsToRun < 1000
        string(Int(model.transactionsToRun))
    elseif model.transactionsToRun < 1_000_000
        string(Int(model.transactionsToRun / 1000)) * " thousand"
    else
        string(Int(model.transactionsToRun / 1_000_000)) * " million"
    end
end


function transactionsToRunStringShort(model::Model)::String
    if model.transactionsToRun < 1000
        model.transactionsToRun
    elseif model.transactionsToRun <1_000_000
        string(Int(model.transactionsToRun / 1000)) * "k"
    else
        string(Int(model.transactionsToRun / 1_000_000)) * "m"
    end
end
   

######################################################

##### REPORTS #####

######################################################

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

    println("\nSum of money by quantile: " ,round(sum(averageByQuantile), digits =0))

    gini = gini_index(averageByQuantile)
    println("\nGini index: ", round(gini, digits = 2))
    entropy = wealth_entropy(averageByQuantile)
    println("Entropy: ", round(entropy, digits = 2))

    println("\nExamine fit with Boltzmann-Gibbs distribution:")
    ratios = successiveRatios(reverse(averageByQuantile))
    println("Inter percentile ratios:\n")
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

################################################################
#                   DSL for constructing a Model
################################################################

function str_to_dict(str::String)
    # Initialize an empty dictionary
    d = Dict{String, String}()

    # Split the string by commas to get the individual key-value pairs
    pairs = split(str, ", ")

    if pairs == []
        return d
    end

    # Iterate through each pair and populate the dictionary
    for pair in pairs
        key, value = split(pair, " = ")
        d[key] = value
    end
    
    return d
end

abstract type Maybe{T} end

struct Just{T} <: Maybe{T}
    value::T
end

struct NothingMaybe <: Maybe{Nothing}
end

just(x) = Just(x)
nothing_maybe() = NothingMaybe()

function maybe_parse_int(s::String)  # Removed the type annotation
    parsed = tryparse(Int64, s)
    return parsed === nothing ? nothing_maybe() : just(parsed)
end

function setNumberOfAgents!(n::String, model::Model)
    println("setNumberOfAgents!: ", n)
    result = maybe_parse_int(n)
    if isa(result, Just)
        model.numberOfAgents = result.value
    else
        println("Bad format for numberOfAgents")
    end
end

function setTransactionsToRun!(nAsString::String, model::Model)
    result = maybe_parse_int(nAsString)
    if isa(result, Just)
        model.transactionsToRun = result.value
    else
        println("Bad format for numberOfTransactionsToRun")
    end
end

function setNumberOfQuantiles!(n::String, model::Model)
    result = maybe_parse_int(n)
    if isa(result, Just)
        model.numberOfQuantiles = result.value
    else
        println("Bad format for numberOfQuantiles")
    end
end

function setShowplot!(showPlotAsString::String, model::Model)
    if showPlotAsString == "true"
        model.showPlot = true
    else
        model.showPlot = false
    end
end

function setPrintModel!(sprintModel::String, model::Model)
    if sprintModel == "true"
        model.printModel = true
    else
        model.printModel = false
    end
end

# Create an empty dictionary to store functions
function_dict = Dict{String, Function}()

# Populate the dictionary with functions
function_dict["agents"] = (model, numberOfAgents) -> setNumberOfAgents!(numberOfAgents, model)
function_dict["transactions"] = (model, transactions) -> setTransactionsToRun!(transactions, model)
function_dict["quantiles"] = (model, quantiles) -> setNumberOfQuantiles!(quantiles, model)
function_dict["plot"] = (model, plot) -> setShowplot!(plot, model)
function_dict["printModel"] = (model, printModel) -> setPrintModel!(printModel, model)


curry(f, x) = (y) -> f(x, y)

function setModel(settings::String)::Model
    settings_dict = str_to_dict(settings)
    model = Model()
    for (key, value) in settings_dict
        try 
            function_dict[key](model, value)
        catch e
            println("Not found: ", e)
        end
    end 
    return model
end
  

# str_to_Model(str::String)::Model
#   dict = str_to_dict(str)
#   for key in keys(dict)
#     if key == "initialBalance"



############ ############ ############ ############ ############ 
#                 INPUT, COMPUTATION, AND OUTPUT
############ ############ ############ ############ ############ 

model = Model(
    numberOfAgents = 1000
    , Δm = 1.0
    , initialBalance = 10
    , transactionsToRun = 1_000_000
    , numberOfQuantiles = 100
    , makeTransaction = makeSimpleTransaction
    , socialSecurityTaxRate = 0.01
    , report = onelineReport #  standardReport # briefReport # onelineReport
    , reportInterval = 1000
    , showEndReport=true
    , showPlot=true
    , verbose=false
    )


# runN(model)

# run("agents = 200, transactions = 1000")

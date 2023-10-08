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
    makeTransaction::Function
end


# Named-argument constructor with default values
function Model(;xMax=100.0, yMax=100.0, initialBalance=10.0, numberOfAgents=100, Δm=1.0, transactionsToRun=1000, makeTransaction=makeSimpleTransaction)
    return Model(xMax, yMax, initialBalance, numberOfAgents, Δm, transactionsToRun, makeTransaction)
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

run = function(model::Model)  # ::Vector{Agent}
  state = init(model)
  return runTransactions(model.transactionsToRun, state)
end

printBalances = function(state)
    balances = sort(getBalances(state::Vector{Agent}))
    for i in 1:model.numberOfAgents
        println(i, ": ", balances[i])
    end
end


########### INPUT, COMPUTATION, AND OUTPUT ########### 

model = Model( makeTransaction = makeSimpleTransaction)
state = run(model)
printBalances(state)


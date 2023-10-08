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
end

model = Model(100.0, 100.0, 10.0, 100, 1.0, 10_000)


# Define the struct for your "thing"
mutable struct Agent
    i:: Int64
    x::Float64
    y::Float64
    balance::Float64
end

# Create an array of "Agent" objects
state = Vector{Agent}(undef, model.numberOfAgents)  # Initialize an empty array of `Thing`

# Populate the array
for i in 1:model.numberOfAgents
    state[i] = Agent(i, model.xMax*rand(), model.yMax*rand(),  model.initialBalance)
end


function distinctRandomPair()
    (i,j) = rand(1:model.numberOfAgents, 2)
    while i == j
      (i,j,)= rand(1:model.numberOfAgents, 2)
    end
    return (i,j)
end

function makeTransaction(state)
    i, j = distinctRandomPair()
    agentA = state[i]
    agentB = state[j]
    if agentA.balance - model.Δm >= 0
        agentA.balance -= model.Δm
        agentB.balance += model.Δm
    end
    return state
end

function run(n::Int64, state::Vector{Agent})::Vector{Agent}
    for i in 1:n
        makeTransaction(state)
    end
    return(state)
end

function getBalances(agents::Vector{Agent})::Vector{Float64}
    return map(agent -> agent.balance, agents)
end

## OUTPUT

# state = makeTransaction(state)
state = run(model.transactionsToRun, state)

state 

println("\nRun makeTransactions: ", model.transactionsToRun,"\n")

balances = sort(getBalances(state))
for i in 1:model.numberOfAgents
    println(i, ": ", balances[i])
end

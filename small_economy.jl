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
end

model = Model(100.0, 100.0, 10.0, 10)


# Define the struct for your "thing"
struct Agent
    i:: Int64
    x::Float64
    y::Float64
    balance::Float64
end

# Create an array of "Agent" objects
state = Vector{Agent}(undef, model.numberOfAgents)  # Initialize an empty array of `Thing`

# Populate the array
for i in 1:model.numberOfAgents
    state[i] = Agent(i, model.xMax*rand(), model.yMax*rand(), rand() * model.initialBalance)
end

# Display the array
for i in 1:model.numberOfAgents
    println( state[i])
end

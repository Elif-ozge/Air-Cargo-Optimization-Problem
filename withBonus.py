#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time

# Load datasets
excel_file = "./END395_ProjectPartIDataset.xlsx"
w_dow = 110257  # Aircraft's dry operating weight in kg
i_doi = 61.6

excel_file2 = "./END395_ProjectPartIIDataset.xlsx"

# Read the Excel files
positions_df = pd.read_excel(excel_file, sheet_name='Positions')
pallets_df = pd.read_excel(excel_file2, sheet_name='Pallets6')
cumulatives_df = pd.read_excel(excel_file2, sheet_name='Position-CG')



def parse_type(x):
    if len(x) > 2 and (x[2] == 'R' or x[2] == 'L') and x[4:6] == '88':
        return 1
    elif len(x) > 2 and (x[2] == 'R' or x[2] == 'L') and x[4:6] == '96':
        return 2
    elif len(x) == 2:
        return 3
    elif len(x) == 1:
        return 4
    elif len(x) > 2 and x[2] == 'P' and x[4:6] == '96':
        return 5
    else:
        return 6


# Identify the decks and if they are aft or front
positions_df['Deck'] = positions_df['Position'].apply(lambda x: 'Lower' if len(x) >= 3 and x[2] == 'P' else 'Main')

# Identify the position parts if they are aft or front
positions_df['Part'] = positions_df.apply(
    lambda x: 'front' if x['H-arm'] < positions_df[positions_df['Position'] == 'J']['H-arm'].values[0] else 'aft',
    axis=1)

# Filter positions for Lower Deck
lower_deck_positions_df = positions_df[positions_df['Deck'] == 'Lower']

# Filter positions for Main Deck
main_deck_positions_df = positions_df[positions_df['Deck'] == 'Main']

#find SBS and SR positions
positions_df['LoadType'] = positions_df['Position'].apply(lambda x: 'SBS' if len(x) >= 3 and x[2] != 'P' else 'SR')

# Filter positions for SBS
sbs_positions_df = positions_df[positions_df['LoadType'] == 'SBS']

# Filter positions for SR
sr_positions_df = positions_df[positions_df['LoadType'] == 'SR']

# Identify the pallet sizes
pallets_df['Size'] = pallets_df['Code'].apply(lambda x: 96 if x[1] == 'M' else 88)
# Identify the position sizes
positions_df['Size'] = positions_df['Position'].apply(
    lambda x: 88 if len(x) == 1 or (len(x) > 3 and x[4:6] == '88') else 96)

positions_df['Type'] = positions_df['Position'].apply(lambda x: parse_type(x))


# Find the set of overlapping positions
def find_overlapping_positions(positions_df):
    overlapping_positions = set()
    for i, row1 in positions_df.iterrows():
        for j, row2 in positions_df.iterrows():

            # Check if positions are on the same deck
            if row1['Deck'] == row2['Deck'] and row1['Type'] != row2['Type'] and row1['Lock1(m)'] <= row2['Lock1(m)']:
                # Check if the positions overlap
                if (row1['Lock2(m)'] >= row2['Lock1(m)']):
                    # Add both combinations since overlap is bidirectional
                    overlapping_positions.add((row1['Position'], row2['Position']))
                    overlapping_positions.add((row2['Position'], row1['Position']))

    return overlapping_positions


overlapping_positions = find_overlapping_positions(positions_df)

# To check if overlapping positions are correct
# s=list(overlapping_positions)
# dict_1 = dict()
# for pos1, pos2 in s:
#     dict_1.setdefault(pos1, []).append(pos2)
# print(dict_1)


# Initialize the model
model = ConcreteModel()

# Sets for pallets and positions
model.pallets = Set(initialize=pallets_df['Code'].tolist())
model.positions = Set(initialize=cumulatives_df['Position'].tolist())

front_positions =positions_df[(positions_df['Part'] == 'front') & (positions_df['Cumulative'] != 0)].sort_values(by='H-arm')['Position'].tolist()
aft_positions =positions_df[(positions_df['Part'] == 'aft') & (positions_df['Cumulative'] != 0)].sort_values(by='H-arm',ascending=False)['Position'].tolist()

# Parameters
model.weights = Param(model.pallets, initialize=pallets_df.set_index('Code')['Weight'].to_dict())
model.max_weights = Param(model.positions, initialize=positions_df.set_index('Position')['Max Weight'].to_dict())
model.h_arms = Param(model.positions, initialize=positions_df.set_index('Position')['H-arm'].to_dict())
model.coefficients = Param(model.positions, initialize=cumulatives_df.set_index('Position')['Coefficient'].to_dict())

model.cum = [1, 2, 3, 4]
model.precedence = [2, 1, 3, 4]
model.cum_j = Param(initialize=1, mutable=True)  # Initialize cum_j_param with the default value
model.efficiency_score = Param(initialize=1, mutable=True)

front_cumulative_dict = {}
for _, row in cumulatives_df.iterrows():
    position = row['Position']
    if (row[1] == 0): break
    for j in range(1, 5):  # Assuming cumulative weights are in columns '1', '2', '3', '4'
        front_cumulative_dict[(j, position)] = row[j]

aft_cumulative_dict = {}
for _, row in cumulatives_df.iloc[::-1].iterrows():
    position = row['Position']
    if (row[1] == 0): break
    for j in range(1, 5):  # Assuming cumulative weights are in columns '1', '2', '3', '4'
        aft_cumulative_dict[(j, position)] = row[j]


# Print keys and values in the dictionary cumulative_dict
# for key, value in aft_cumulative_dict.items():
#     print(key, value)

def c_front_init(model, i, j):
    return front_cumulative_dict[(i, j)]


def c_aft_init(model, i, j):
    return aft_cumulative_dict[(i, j)]


model.front_cumulatives = Param(model.cum, front_positions, initialize=c_front_init)
model.aft_cumulatives = Param(model.cum, aft_positions, initialize=c_aft_init)

model.pallet_sizes = Param(model.pallets, initialize=pallets_df.set_index('Code')['Size'].to_dict())
model.position_sizes = Param(model.positions, initialize=positions_df.set_index('Position')['Size'].to_dict())

# Overlapping Positions Parameter
model.overlapping_positions = Param(model.positions, initialize=overlapping_positions)

# Define parameters in for DOW and DOI
model.w_dow = Param(initialize=w_dow)
model.i_doi = Param(initialize=i_doi)

# Decision Variables
model.I = RangeSet(4)
model.x = Var(model.pallets, model.positions, within=pyomo.environ.Binary)
model.y = Var(model.I, within=pyomo.environ.Binary)


def Type_Restriction(pallet):
    if 'RestrictedLoadingType' in pallets_df.columns:
        # Get the restricted load type for the current pallet
        restricted_load_type = pallets_df.loc[pallets_df['Code'] == pallet, 'RestrictedLoadingType'].values
        # Check if a restricted load type is defined for the current pallet
        if len(restricted_load_type) > 0 and restricted_load_type[0] != 0:
            # Find suitable positions based on the restricted load type
            suitable_positions = [position for position in model.positions if positions_df.loc[positions_df['Position'] == position, 'LoadType'].values[0] == restricted_load_type[0]]
            # If suitable positions are found, enforce that the pallet is assigned to exactly one of them
            if len(suitable_positions) > 0:
                return sum(model.x[pallet, position] for position in suitable_positions) == 1
        # If no suitable position is found or no restricted load type is defined, return no constraint
        return Constraint.Skip
    else:
        # If the 'RestrictedLoadingType' column does not exist, return no constraint
        return Constraint.Skip

# Add constraints for each pallet
model.assign_restricted_load_type_constraints = ConstraintList(rule=(Type_Restriction(pallet) for pallet in model.pallets))


def Position_Restriction(pallet):
    if 'RestrictedLoadingType' in pallets_df.columns:
        # Get the restricted load type for the current pallet
        restricted_position = pallets_df.loc[pallets_df['Code'] == pallet, 'RestrictedPosition'].values[0]
        if restricted_position !=0:
            return model.x[pallet, restricted_position] == 1
        else:
            return Constraint.Skip
    else:
        # If the 'RestrictedLoadingType' column does not exist, return no constraint
        return Constraint.Skip

# Add constraints for each pallet
model.assign_specified_position_type_constraints = ConstraintList(rule=(Position_Restriction(pallet) for pallet in model.pallets))

def Location_Restriction(pallet):
    if 'RestrictedLocation' in pallets_df.columns:
        # Get the restricted load type for the current pallet
        restricted_location = pallets_df.loc[pallets_df['Code'] == pallet, 'RestrictedLocation'].values[0]
       # Check if the pallet is placed on the expected deck
        if restricted_location == 'LD':
            return sum(model.x[pallet, position] for position in lower_deck_positions_df['Position']) == 1
        elif restricted_location == 'M':
            return sum(model.x[pallet, position] for position in main_deck_positions_df['Position']) == 1
        else:
            # If the expected deck is not specified or invalid, return no constraint
            return Constraint.Skip
    else:
        # If the 'Deck' column does not exist, return no constraint
        return Constraint.Skip

# Add constraints for each pallet
model.assign_specified_position_type_constraints = ConstraintList(rule=(Location_Restriction(pallet) for pallet in model.pallets))



def PAG_compatibility_constraint(model, pallet, position):
    # If the position is for 96" pallets, PAG pallets cannot be placed
    if model.pallet_sizes[pallet] == 88 and model.position_sizes[position] == 96:
        return model.x[pallet, position] == 0
    elif model.pallet_sizes[pallet] == 96 and model.position_sizes[position] == 88:
        return model.x[pallet, position] == 0
    else:
        return Constraint.Skip

# Apply the constraint only for PAG pallets and all positions
model.PAGCompatibility = Constraint(model.pallets, model.positions, rule=PAG_compatibility_constraint)


# Weight Limit for Each Position
def weight_limit_rule(model, position):
    return sum(model.x[pallet, position] * model.weights[pallet] for pallet in model.pallets) <= model.max_weights[
        position]


model.weight_limit_constraints = Constraint(model.positions, rule=weight_limit_rule)


def cumulative_weight_rule_front(model, positions_list):
    for i, position in enumerate(positions_list):
        # Calculate cumulative weight up to and including the current position
        weight_sum = sum(
            model.x[pallet, positions_list[j]] * model.weights[pallet] * model.coefficients[positions_list[j]]
            for j in range(i + 1)  # From the first position up to and including the current position
            for pallet in model.pallets)

        # Constraint: cumulative weight must not exceed the limit for the current position
        yield weight_sum <= sum(model.front_cumulatives[i, position] * model.y[i] for i in model.I)


def cumulative_weight_rule_aft(model, positions_list):
    for i, position in enumerate(positions_list):
        # Calculate cumulative weight up to and including the current position
        weight_sum = sum(
            model.x[pallet, positions_list[j]] * model.weights[pallet] * model.coefficients[positions_list[j]]
            for j in range(i + 1)  # From the first position up to and including the current position
            for pallet in model.pallets)

        # Constraint: cumulative weight must not exceed the limit for the current position
        yield weight_sum <= sum(model.aft_cumulatives[i, position] * model.y[i] for i in model.I)

model.cumulative_weight_constraints_front_main = ConstraintList(
    rule=(rule for rule in cumulative_weight_rule_front(model, front_positions)))
model.cumulative_weight_constraints_aft_main = ConstraintList(
    rule=(rule for rule in cumulative_weight_rule_aft(model, aft_positions)))


# Position Unique Assignment
def position_unique_assignment_rule(model, position):
    return sum(model.x[pallet, position] for pallet in model.pallets) <= 1


model.position_unique_assignment = Constraint(model.positions, rule=position_unique_assignment_rule)


# Unique Assignment for Each Pallet
def unique_assignment_rule(model, pallet):
    return sum(model.x[pallet, position] for position in model.positions) == 1


model.unique_assignment_constraints = Constraint(model.pallets, rule=unique_assignment_rule)


# Blue Envelope Constraint
def calc_position_index(model, pallet, position):
    return (model.h_arms[position] - 36.3495) * model.weights[pallet] * model.x[pallet, position] / 2500


# Total weight and index for the aircraft
def total_weight(model):
    return total_weight_aft(model) + total_weight_front(model) + model.w_dow


def total_weight_front(model):
    return sum(
        model.x[pallet, position] * model.weights[pallet] for pallet in model.pallets for position in front_positions)


def total_weight_aft(model):
    return sum(
        model.x[pallet, position] * model.weights[pallet] for pallet in model.pallets for position in aft_positions)


def total_index(model):
    return total_index_aft(model) + total_index_front(model) + model.i_doi


def total_index_front(model):
    return sum(calc_position_index(model, pallet, position) for pallet in model.pallets for position in front_positions)


def total_index_aft(model):
    return sum(calc_position_index(model, pallet, position) for pallet in model.pallets for position in aft_positions)


# Define the constraints based on the Blue Envelope inequalities
def blue_envelope_lower_weight_constraint(model):
    return total_weight(model) >= 120000


def blue_envelope_upper_weight_constraint(model):
    return total_weight(model) <= 180000


def blue_envelope_lower_boundary_constraint(model):
    return total_weight(model) >= 2000 * total_index(model) - 240000


def blue_envelope_upper_boundary_constraint(model):
    return total_weight(model) >= -1000 * total_index(model) + 235000


# Adding constraints to the model
model.blue_envelope_lower_weight = Constraint(rule=blue_envelope_lower_weight_constraint)
model.blue_envelope_upper_weight = Constraint(rule=blue_envelope_upper_weight_constraint)
model.blue_envelope_lower_boundary = Constraint(rule=blue_envelope_lower_boundary_constraint)
model.blue_envelope_upper_boundary = Constraint(rule=blue_envelope_upper_boundary_constraint)

# Only one interval choosen
model.unique_interval_rule = Constraint(expr=sum(model.y[i] for i in model.I) == 1)


# Overlapping_positions is a set of tuples with the overlapping position pairs
# Add the overlapping constraint to the model
def overlapping_positions_rule(model, pos1, pos2):
    # The constraint states that the sum of any two overlapping positions cannot exceed 1
    # across all pallets, meaning two pallets cannot occupy overlapping positions simultaneously
    return sum(model.x[pallet, pos1] for pallet in model.pallets) + sum(
        model.x[pallet, pos2] for pallet in model.pallets) <= 1


# Create a constraint for each pair of overlapping positions
for pos1, pos2 in overlapping_positions:
    # Generate a unique name for each constraint based on the position pair
    constraint_name = f"overlap_{pos1}_{pos2}"
    model.add_component(constraint_name, Constraint(rule=lambda m: overlapping_positions_rule(m, pos1, pos2)))

# Objective function to optimizes the fuel consuption
scores = {1: 3, 2: 4, 3: 2, 4: 1}
model.obj = Objective(expr=sum(scores[i] * model.y[i] for i in model.I), sense=maximize)

start_time = time.time()

# Solve the model
solver = SolverFactory('cplex')
solution = solver.solve(model)

cpu_time = time.time() - start_time
# Specify the path to your output text file
output_file_path = './model_output.txt'

# Open the file in write mode ('w')
with open(output_file_path, 'w') as file:
    # Redirect the standard output to the file
    model.pprint(ostream=file)

# Print the solution
for pallet in model.pallets:
    for position in model.positions:
        if model.x[pallet, position].value > 0.5:  # Assuming a binary variable, adjust threshold if necessary
            print(f"Pallet {pallet} loaded onto position {position}.")
print("CPU Time:", cpu_time, "seconds")
print("CPU Time:", value(model.obj), "")


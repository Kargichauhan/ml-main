import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import matplotlib.ticker as ticker
from functools import partial
import pandas as pd


class station_item:
    def __init__(self):
        self.status = "healthy"  # can be either "healthy", "damaged", or "repaired"
        self.location = "active"  # can be either "active" or "warehouse"
        self.repair_attempted = 0


class instrument_1(station_item):  # Wide Angle Camera
    def __init__(self):
        station_item.__init__(self)
        self.name = "Instrument_1"
        self.failure_probability = 1 / 365  # 1 failure per year assumption
        self.repair_probability = 0.01  # 1% chance of repair


class instrument_2(station_item):  # Medium Range Camera
    def __init__(self):
        station_item.__init__(self)
        self.name = "Instrument_2"
        self.failure_probability = 1 / (3 * 365)  # 1 failure per 3 years assumption
        self.repair_probability = 0.01  # 1% chance of repair


class instrument_3(station_item):  # Telephoto Camera
    def __init__(self):
        station_item.__init__(self)
        self.name = "Instrument_3"
        self.failure_probability = 1 / (5 * 365)  # 1 failure per 5 years assumption
        self.repair_probability = 0.01  # 1% chance of repair


class TTC_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "TTC_Unit"
        self.failure_probability = 1 / (0.5 * 365)  # 1 failure per 6 months assumption
        self.repair_probability = 0.01  # 1% chance of repair


class ADC_GNC_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "ADC_GNC_Unit"
        self.failure_probability = 1 / (0.25 * 365)  # 1 failure per 3 months assumption
        self.repair_probability = 0.01  # 1% chance of repair


class OBC_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "OBC_Unit"
        self.failure_probability = 1 / 60  # 1 failure per 2 months assumption
        self.repair_probability = 0.01  # 1% chance of repair


class TCS_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "TCS_Unit"
        self.failure_probability = 1 / 365  # 1 failure per year assumption
        self.repair_probability = 0.05  # 5% chance of repair


class truss_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "Truss_Unit"
        self.failure_probability = 1 / 30  # 1 failure per month assumption
        self.repair_probability = 0.30  # 30% chance of repair


class EPS_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "EPS_Unit"
        self.failure_probability = 1 / 30  # 1 failure per year assumption
        self.repair_probability = 0.01  # 1% chance of repair


class robot_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "Robot_Unit"
        self.failure_probability = 1 / (3 * 365)  # 1 failure per year assumption
        self.repair_probability = 0.30  # 30% chance of repair


class payload_support_unit(station_item):
    def __init__(self):
        station_item.__init__(self)
        self.name = "Payload_Support_Unit"
        self.failure_probability = 1 / (0.25 * 365)  # 1 failure per year assumption
        self.repair_probability = 0.01  # 1% chance of repair


def Station_Inventory_Simulation(no_of_days):
    """
    This function runs the simulation for the specified number of days and then returns a DataFrame of the following spec:
    {
        "Day": list_of_days,
        "Total Inventory": list_of_total_inventory,
        "Number of Active Inventory": list_of_active_inventory,
        "Number of Warehouse Inventory":list_of_warehouse_inventory,
        "Number of Healthy Inventory": list_of_healthy_inventory,
        "Number of Damaged Inventory": list_of_damaged_inventory,
        "Number of Damages": list_of_damages,
        "Number of Repairs": list_of_repairs,
    }
    The parameters that can be modified within the function are:
    number_of_instrument_1
    number_of_instrument_2
    number_of_instrument_3
    number of other station elements
    number of spares
    whether you want one repair attempt or multiple repair attempts per damaged element per simulation
    """
    total_inventory = []
    healthy_inventory = []  # status
    damaged_inventory = []  # status
    active_inventory = []  # location
    warehouse_inventory = []  # location

    no_of_instrument_1 = 4 * 3  # From CAD Model
    no_of_instrument_2 = 12 * 3  # From CAD model
    no_of_instrument_3 = 28 * 3  # From CAD Model
    no_of_payload_support = (
        no_of_instrument_1 + no_of_instrument_2 + no_of_instrument_3
    )  # Assumption

    no_of_TTC = 40 * 3  # Assumption
    no_of_ADC_GNC = 40 * 3  # Assumption
    no_of_OBC = 20 * 3  # Assumption
    no_of_TCS = 50 * 3  # Assumption
    no_of_truss = 1000 * 3  # Assumption
    no_of_EPS = 50 * 3  # Assumption
    no_of_robot = 5 * 3  # Assumption

    ####################### Spawning the Active Healthy components ######################

    for i in range(1, no_of_instrument_1 + 1):
        ins_1 = instrument_1()
        # ins_1.name = ins_1.name + " " + str(i)
        total_inventory.append(ins_1)
        healthy_inventory.append(ins_1)
        active_inventory.append(ins_1)

    for i in range(1, no_of_instrument_2 + 1):
        ins_2 = instrument_2()
        # ins_2.name = ins_2.name + " " + str(i)
        total_inventory.append(ins_2)
        healthy_inventory.append(ins_2)
        active_inventory.append(ins_2)

    for i in range(1, no_of_instrument_3 + 1):
        ins_3 = instrument_3()
        # ins_3.name = ins_3.name + " " + str(i)
        total_inventory.append(ins_3)
        healthy_inventory.append(ins_3)
        active_inventory.append(ins_3)

    for i in range(1, no_of_payload_support + 1):
        psu = payload_support_unit()
        # psu.name = psu.name + " " + str(i)
        total_inventory.append(psu)
        healthy_inventory.append(psu)
        active_inventory.append(psu)

    for i in range(1, no_of_TTC + 1):
        ttc = TTC_unit()
        # ttc.name = ttc.name + " " + str(i)
        total_inventory.append(ttc)
        healthy_inventory.append(ttc)
        active_inventory.append(ttc)

    for i in range(1, no_of_ADC_GNC + 1):
        adc = ADC_GNC_unit()
        # adc.name = adc.name + " " + str(i)
        total_inventory.append(adc)
        healthy_inventory.append(adc)
        active_inventory.append(adc)

    for i in range(1, no_of_OBC + 1):
        obc = OBC_unit()
        # obc.name = obc.name + " " + str(i)
        total_inventory.append(obc)
        healthy_inventory.append(obc)
        active_inventory.append(obc)

    for i in range(1, no_of_TCS + 1):
        tcs = TCS_unit()
        # tcs.name = tcs.name + " " + str(i)
        total_inventory.append(tcs)
        healthy_inventory.append(tcs)
        active_inventory.append(tcs)

    for i in range(1, no_of_truss + 1):
        trs = truss_unit()
        # trs.name = trs.name + " " + str(i)
        total_inventory.append(trs)
        healthy_inventory.append(trs)
        active_inventory.append(trs)

    for i in range(1, no_of_EPS + 1):
        eps = EPS_unit()
        # eps.name = eps.name + " " + str(i)
        total_inventory.append(eps)
        healthy_inventory.append(eps)
        active_inventory.append(eps)

    for i in range(1, no_of_robot + 1):
        rbt = robot_unit()
        # rbt.name = rbt.name + " " + str(i)
        total_inventory.append(rbt)
        healthy_inventory.append(rbt)
        active_inventory.append(rbt)

    ####################### Spawning the Spare Healthy components ######################

    no_of_spares = (
        5  # Assuming 5 spares for each. Need to tune it for individual elements
    )

    for j in range(1, no_of_spares + 1):
        ins_1_spare = instrument_1()
        # ins_1_spare.name = ins_1_spare.name + " spare " + str(j)
        ins_1_spare.location = "warehouse"
        total_inventory.append(ins_1_spare)
        healthy_inventory.append(ins_1_spare)
        warehouse_inventory.append(ins_1_spare)

        ins_2_spare = instrument_2()
        # ins_2_spare.name = ins_2_spare.name + " spare " + str(j)
        ins_2_spare.location = "warehouse"
        total_inventory.append(ins_2_spare)
        healthy_inventory.append(ins_2_spare)
        warehouse_inventory.append(ins_2_spare)

        ins_3_spare = instrument_3()
        # ins_3_spare.name = ins_3_spare.name + " spare " + str(j)
        ins_3_spare.location = "warehouse"
        total_inventory.append(ins_3_spare)
        healthy_inventory.append(ins_3_spare)
        warehouse_inventory.append(ins_3_spare)

        psu_spare = payload_support_unit()
        # psu_spare.name = psu_spare.name + " spare " + str(j)
        psu_spare.location = "warehouse"
        total_inventory.append(psu_spare)
        healthy_inventory.append(psu_spare)
        warehouse_inventory.append(psu_spare)

        ttc_spare = TTC_unit()
        # ttc_spare.name = ttc_spare.name + " spare " + str(j)
        ttc_spare.location = "warehouse"
        total_inventory.append(ttc_spare)
        healthy_inventory.append(ttc_spare)
        warehouse_inventory.append(ttc_spare)

        adc_spare = ADC_GNC_unit()
        # adc_spare.name = adc_spare.name + " spare " + str(j)
        adc_spare.location = "warehouse"
        total_inventory.append(adc_spare)
        healthy_inventory.append(adc_spare)
        warehouse_inventory.append(adc_spare)

        obc_spare = OBC_unit()
        # obc_spare.name = obc_spare.name + " spare " + str(j)
        obc_spare.location = "warehouse"
        total_inventory.append(obc_spare)
        healthy_inventory.append(obc_spare)
        warehouse_inventory.append(obc_spare)

        tcs_spare = TCS_unit()
        # tcs_spare.name = tcs_spare.name + " spare " + str(j)
        tcs_spare.location = "warehouse"
        total_inventory.append(tcs_spare)
        healthy_inventory.append(tcs_spare)
        warehouse_inventory.append(tcs_spare)

        trs_spare = truss_unit()
        # trs_spare.name = trs_spare.name + " spare " + str(j)
        trs_spare.location = "warehouse"
        total_inventory.append(trs_spare)
        healthy_inventory.append(trs_spare)
        warehouse_inventory.append(trs_spare)

        eps_spare = EPS_unit()
        # eps.name = eps.name + " spare " + str(j)
        eps_spare.location = "warehouse"
        total_inventory.append(eps_spare)
        healthy_inventory.append(eps_spare)
        warehouse_inventory.append(eps_spare)

        rbt_spare = robot_unit()
        # rbt_spare.name = rbt_spare.name + " spare " + str(j)
        rbt_spare.location = "warehouse"
        total_inventory.append(rbt_spare)
        healthy_inventory.append(rbt_spare)
        warehouse_inventory.append(rbt_spare)

    ######################## Running the simulation ###########################################

    # Initializing all the below lists to append to each day
    list_of_total_inventory = []
    list_of_active_inventory = []
    list_of_healthy_inventory = []
    list_of_damaged_inventory = []
    list_of_warehouse_inventory = []
    list_of_damages = []
    list_of_repairs = []
    list_of_days = []

    for day in range(no_of_days):
        print(f"{day+1}")

        # Initialize damages and repairs to zero to count for each day separately
        no_of_damages = 0
        no_of_repairs = 0

        # Step 1: Going through all "active" inventory and then checking if they fail today
        for active_element in active_inventory:
            # Doing a binomial weighted coin toss to check if the element fails
            if np.random.binomial(1, active_element.failure_probability) == 1:
                # If it fails, then remove it from the active inventory, move to warehouse, remove from healthy inventory, then set it to damaged
                # Then, add one to the number of damaged elements of the day
                active_element.status = "damaged"
                active_element.location = "warehouse"
                active_inventory.remove(active_element)
                healthy_inventory.remove(active_element)
                damaged_inventory.append(active_element)
                warehouse_inventory.append(active_element)
                no_of_damages += 1

                # If there is a failure, then go through the warehouse, see if there is a healthy element matching the name
                # If there is a match, then move it to the active inventory and remove it from the warehouse
                for warehouse_element in warehouse_inventory:
                    if (warehouse_element.name == active_element.name) & (
                        warehouse_element in healthy_inventory
                    ):
                        warehouse_element.location = "active"
                        active_inventory.append(warehouse_element)
                        warehouse_inventory.remove(warehouse_element)

        # Step 2: Going through the "damaged" inventory and then trying to repair them
        for damaged_element in damaged_inventory:
            # The below condition is to check if there were prior attempts to repair the element
            # It only works if the repair_attempted attribute is set to 1 at the end after the attempt
            if damaged_element.repair_attempted == 0:
                # Doing a binomial weighted coin toss to check if the element is repaired
                if np.random.binomial(1, damaged_element.repair_probability) == 1:
                    # IF successful, set the element to healthy, remove it from the damaged list, and then add it to the healthy list
                    # Then, add one to the number of repaired elements of the day
                    damaged_element.status = "healthy"
                    damaged_inventory.remove(damaged_element)
                    healthy_inventory.append(damaged_element)
                    no_of_repairs += 1
                    damaged_element.repair_attempted = 1  # comment this out if you want unlimited repair attempts for each element per simulation

        list_of_active_inventory.append(len(active_inventory))
        list_of_healthy_inventory.append(len(healthy_inventory))
        list_of_damaged_inventory.append(len(damaged_inventory))
        list_of_warehouse_inventory.append(len(warehouse_inventory))
        list_of_total_inventory.append(len(total_inventory))
        list_of_damages.append(no_of_damages)
        list_of_repairs.append(no_of_repairs)
        list_of_days.append(day + 1)  # for visualizing, not indexing

    df = pd.DataFrame(
        data={
            "Day": np.array(list_of_days),
            "Total Inventory": np.array(list_of_total_inventory),
            "Number of Active Inventory": np.array(list_of_active_inventory),
            "Number of Warehouse Inventory": np.array(list_of_warehouse_inventory),
            "Number of Healthy Inventory": np.array(list_of_healthy_inventory),
            "Number of Damaged Inventory": np.array(list_of_damaged_inventory),
            "Number of Damages": np.array(list_of_damages),
            "Number of Repairs": np.array(list_of_repairs),
        },
    )

    return df


###########################################################################################################################

# Setting up the number of simulations and the number of days per simulation
no_of_simulations = 10
no_of_days = 30

# Initializing DataFrames to store the data from multiple simulations
df_total_inventory = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_active_inventory = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_warehouse_inventory = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_healthy_inventory = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_damaged_inventory = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_damages = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})
df_repairs = pd.DataFrame(data={"Day": np.arange(1, no_of_days + 1)})

df_list = [
    df_total_inventory,
    df_active_inventory,
    df_warehouse_inventory,
    df_healthy_inventory,
    df_damaged_inventory,
    df_damages,
    df_repairs,
]

label_list = [
    "Total Inventory",
    "Active Inventory",
    "Warehouse Inventory",
    "Healthy Inventory",
    "Damaged Inventory",
    "Number of Damages",
    "Number of Repairs",
]  # For the plot function

color_list = [
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "cyan",
    "purple",
]  # For the plot function

for simulation in range(1, no_of_simulations + 1):
    print(f"Simulation = {simulation}")
    df = Station_Inventory_Simulation(no_of_days=no_of_days)

    df_total_inventory[f"Total Inventory {simulation}"] = df["Total Inventory"]
    df_active_inventory[f"Active Inventory {simulation}"] = df[
        "Number of Active Inventory"
    ]
    df_warehouse_inventory[f"Warehouse Inventory {simulation}"] = df[
        "Number of Warehouse Inventory"
    ]
    df_healthy_inventory[f"Healthy Inventory {simulation}"] = df[
        "Number of Healthy Inventory"
    ]
    df_damaged_inventory[f"Damaged Inventory {simulation}"] = df[
        "Number of Damaged Inventory"
    ]
    df_damages[f"Damaged Inventory {simulation}"] = df["Number of Damages"]
    df_repairs[f"Repaired Inventory {simulation}"] = df["Number of Repairs"]


fig1 = plt.figure(figsize=(12, 12))
fig1.canvas.manager.set_window_title("Station Inventory Simulations")  # type: ignore
ax = fig1.add_subplot()
ax.set_xlabel("Days")
ax.set_ylabel("Station Elements")

for i in range(len(df_list)):
    df = df_list[i]

    df["Mean"] = df.iloc[:, 1:].mean(axis=1, numeric_only=True).round(decimals=3)
    df["STD"] = df.iloc[:, 1:].std(axis=1, numeric_only=True).round(decimals=3)
    # https://stackoverflow.com/a/22149930

    ax.errorbar(
        df["Day"],
        df["Mean"],
        yerr=df["STD"],
        label=label_list[i],
        color=color_list[i],
        capsize=5,
    )


ax.legend()

plt.show()

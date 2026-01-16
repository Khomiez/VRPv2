"""
Create an Excel template for VRP solver input data using xlsxwriter.
"""
import xlsxwriter


def create_vrp_template(output_path: str = "vrp_input_template.xlsx"):
    """Create an Excel template for VRP solver input."""
    wb = xlsxwriter.Workbook(output_path)

    # Define formats
    header_format = wb.add_format({
        'bold': True,
        'font_color': 'white',
        'bg_color': '#4472C4',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })

    example_format = wb.add_format({
        'bg_color': '#E2EFDA',
        'border': 1
    })

    example_center = wb.add_format({
        'bg_color': '#E2EFDA',
        'border': 1,
        'align': 'center'
    })

    required_format = wb.add_format({
        'bg_color': '#FCE4D6',
        'border': 1
    })

    normal_format = wb.add_format({'border': 1})

    title_format = wb.add_format({'bold': True, 'font_size': 14})
    section_format = wb.add_format({'bold': True, 'font_color': '#4472C4'})
    warning_format = wb.add_format({'bold': True, 'font_color': 'red'})

    # ==================== Sheet 1: Instructions ====================
    ws_inst = wb.add_worksheet("Instructions")
    ws_inst.set_column('A:A', 60)

    instructions = [
        ("VRP Solver Input Template", title_format),
        ("", None),
        ("This template has 4 data sheets:", None),
        ("1. Nodes - Location information", None),
        ("2. Vehicles - Vehicle specifications", None),
        ("3. Distance_Matrix - Distances between nodes (meters)", None),
        ("4. Settings - Depot configuration", None),
        ("", None),
        ("=== NODES SHEET ===", section_format),
        ("Node ID: Unique integer starting from 1", None),
        ("Node Name: Descriptive name for the location", None),
        ("General Demand: Amount of general waste to collect (units)", None),
        ("Recycle Demand: Amount of recyclable waste to collect (units)", None),
        ("Node Type: 'depot', 'checkpoint', or 'customer'", None),
        ("", None),
        ("IMPORTANT RULES:", warning_format),
        ("- Node ID 1 MUST be the depot (starting point)", None),
        ("- At least one node must be type 'checkpoint'", None),
        ("- Depot and checkpoint should have 0 demand", None),
        ("", None),
        ("=== VEHICLES SHEET ===", section_format),
        ("Vehicle Type: Identifier (A, B, C, etc.)", None),
        ("General Capacity: Max general waste capacity (units)", None),
        ("Recycle Capacity: Max recyclable waste capacity (units)", None),
        ("Fixed Cost: Cost per vehicle deployment (THB)", None),
        ("Fuel Cost: Cost per kilometer traveled (THB/km)", None),
        ("", None),
        ("=== DISTANCE MATRIX SHEET ===", section_format),
        ("- Enter distances between ALL nodes in METERS", None),
        ("- Matrix must be symmetric (i to j = j to i)", None),
        ("- Diagonal must be 0", None),
        ("", None),
        ("=== HOW TO USE ===", section_format),
        ("1. Fill in Nodes, Vehicles, and Distance_Matrix sheets", None),
        ("2. Green rows are examples - replace with your data", None),
        ("3. Save the file", None),
        ("4. Run: python convert_template_to_json.py <this_file.xlsx>", None),
        ("5. Use the generated JSON with the VRP solver", None),
    ]

    for row, (text, fmt) in enumerate(instructions):
        if fmt:
            ws_inst.write(row, 0, text, fmt)
        else:
            ws_inst.write(row, 0, text)

    # ==================== Sheet 2: Nodes ====================
    ws_nodes = wb.add_worksheet("Nodes")

    # Set column widths
    ws_nodes.set_column('A:A', 12)
    ws_nodes.set_column('B:B', 20)
    ws_nodes.set_column('C:C', 18)
    ws_nodes.set_column('D:D', 18)
    ws_nodes.set_column('E:E', 15)

    # Headers
    node_headers = ["Node ID", "Node Name", "General Demand", "Recycle Demand", "Node Type"]
    for col, header in enumerate(node_headers):
        ws_nodes.write(0, col, header, header_format)

    # Example data
    example_nodes = [
        [1, "Depot", 0, 0, "depot"],
        [2, "Location A", 50, 5, "customer"],
        [3, "Location B", 80, 10, "customer"],
        [4, "Location C", 30, 3, "customer"],
        [5, "Checkpoint", 0, 0, "checkpoint"],
    ]

    for row, node_data in enumerate(example_nodes, 1):
        for col, value in enumerate(node_data):
            ws_nodes.write(row, col, value, example_format)

    # ==================== Sheet 3: Vehicles ====================
    ws_vehicles = wb.add_worksheet("Vehicles")

    # Set column widths
    for col in range(5):
        ws_vehicles.set_column(col, col, 18)

    # Headers
    vehicle_headers = ["Vehicle Type", "General Capacity", "Recycle Capacity", "Fixed Cost (THB)", "Fuel Cost (THB/km)"]
    for col, header in enumerate(vehicle_headers):
        ws_vehicles.write(0, col, header, header_format)

    # Example data
    example_vehicles = [
        ["A", 2000, 200, 2400, 8],
        ["B", 2000, 200, 2400, 8.5],
        ["C", 2000, 200, 2400, 9],
    ]

    for row, vehicle_data in enumerate(example_vehicles, 1):
        for col, value in enumerate(vehicle_data):
            ws_vehicles.write(row, col, value, example_format)

    # ==================== Sheet 4: Distance Matrix ====================
    ws_dist = wb.add_worksheet("Distance_Matrix")

    # Set column widths
    ws_dist.set_column('A:A', 12)
    for col in range(1, 20):
        ws_dist.set_column(col, col, 10)

    # Corner cell
    ws_dist.write(0, 0, "From / To", header_format)

    # Header row (node IDs)
    for col in range(1, 6):
        ws_dist.write(0, col, col, header_format)

    # Example distance matrix (5x5, symmetric, in meters)
    example_distances = [
        [0, 1500, 2300, 1800, 3000],
        [1500, 0, 1200, 2100, 2500],
        [2300, 1200, 0, 900, 1800],
        [1800, 2100, 900, 0, 2200],
        [3000, 2500, 1800, 2200, 0],
    ]

    for row, row_data in enumerate(example_distances, 1):
        # Row header (node ID)
        ws_dist.write(row, 0, row, header_format)
        # Distance values
        for col, distance in enumerate(row_data, 1):
            ws_dist.write(row, col, distance, example_center)

    # Note about units
    ws_dist.write(7, 0, "Note: All distances must be in METERS", warning_format)

    # ==================== Sheet 5: Settings ====================
    ws_settings = wb.add_worksheet("Settings")

    # Set column widths
    ws_settings.set_column('A:A', 20)
    ws_settings.set_column('B:B', 15)
    ws_settings.set_column('C:C', 35)

    # Headers
    settings_headers = ["Setting", "Value", "Description"]
    for col, header in enumerate(settings_headers):
        ws_settings.write(0, col, header, header_format)

    # Settings data
    settings_data = [
        ["Depot Node ID", 1, "ID of the depot node (usually 1)"],
        ["Depot Name", "Depot", "Display name for the depot"],
    ]

    for row, (setting, value, desc) in enumerate(settings_data, 1):
        ws_settings.write(row, 0, setting, normal_format)
        ws_settings.write(row, 1, value, required_format)
        ws_settings.write(row, 2, desc, normal_format)

    wb.close()
    print(f"Template created: {output_path}")


if __name__ == "__main__":
    create_vrp_template("vrp_input_template.xlsx")

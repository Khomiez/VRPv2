# -*- coding: utf-8 -*-
"""
VRP Solver v2 - Visualization Generator

Generates visualizations for all VRP solutions:
1. Cost breakdown charts (pie and bar)
2. Capacity utilization charts
3. Route visualization
4. Summary comparison charts
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import math

# Set Thai font support and style
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'accent': '#FF9800',
    'fixed_cost': '#E91E63',
    'fuel_cost': '#00BCD4',
    'general': '#3F51B5',
    'recycle': '#8BC34A',
    'route1': '#2196F3',
    'route2': '#FF5722',
    'depot': '#F44336',
    'checkpoint': '#9C27B0',
    'node': '#607D8B'
}


def load_solution(file_path: str) -> dict:
    """Load solution JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_cost_breakdown_chart(solution: dict, sheet_name: str, output_dir: Path):
    """Create cost breakdown pie and bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data
    fixed_cost = solution['total_fixed_cost']
    fuel_cost = solution['total_fuel_cost']
    total_cost = solution['total_cost']

    # Pie chart
    ax1 = axes[0]
    sizes = [fixed_cost, fuel_cost]
    labels = [f'Fixed Cost\n({fixed_cost:,.0f} THB)', f'Fuel Cost\n({fuel_cost:,.2f} THB)']
    colors = [COLORS['fixed_cost'], COLORS['fuel_cost']]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90,
                                        textprops={'fontsize': 11})
    ax1.set_title(f'Cost Breakdown - {sheet_name} Zones\nTotal: {total_cost:,.2f} THB',
                  fontsize=14, fontweight='bold')

    # Bar chart for routes
    ax2 = axes[1]
    routes = solution['routes']
    x = np.arange(len(routes))
    width = 0.35

    fixed_costs = [r['fixed_cost'] for r in routes]
    fuel_costs = [r['fuel_cost'] for r in routes]

    bars1 = ax2.bar(x - width/2, fixed_costs, width, label='Fixed Cost', color=COLORS['fixed_cost'])
    bars2 = ax2.bar(x + width/2, fuel_costs, width, label='Fuel Cost', color=COLORS['fuel_cost'])

    ax2.set_xlabel('Vehicle', fontsize=12)
    ax2.set_ylabel('Cost (THB)', fontsize=12)
    ax2.set_title(f'Cost by Vehicle - {sheet_name} Zones', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Vehicle {r["vehicle_id"]}' for r in routes])
    ax2.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:,.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / f'cost_breakdown_{sheet_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_file.name}')


def create_capacity_chart(solution: dict, sheet_name: str, output_dir: Path):
    """Create capacity utilization chart."""
    routes = solution['routes']
    num_routes = len(routes)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(num_routes)
    width = 0.35

    # Capacity limits
    general_capacity = 2000
    recycle_capacity = 200

    # Data
    general_loads = [r['general_load'] for r in routes]
    recycle_loads = [r['recycle_load'] for r in routes]
    general_pct = [load / general_capacity * 100 for load in general_loads]
    recycle_pct = [load / recycle_capacity * 100 for load in recycle_loads]

    bars1 = ax.bar(x - width/2, general_pct, width, label='General Waste', color=COLORS['general'])
    bars2 = ax.bar(x + width/2, recycle_pct, width, label='Recycle Waste', color=COLORS['recycle'])

    # Add 100% line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Capacity Limit (100%)')

    ax.set_xlabel('Vehicle', fontsize=12)
    ax.set_ylabel('Capacity Utilization (%)', fontsize=12)
    ax.set_title(f'Capacity Utilization - {sheet_name} Zones', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Vehicle {r["vehicle_id"]}\n({r["vehicle_type"]})' for r in routes])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 120)

    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.annotate(f'{general_loads[i]:,.0f}\n({height:.1f}%)',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.annotate(f'{recycle_loads[i]:,.0f}\n({height:.1f}%)',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / f'capacity_{sheet_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_file.name}')


def create_route_visualization(solution: dict, sheet_name: str, output_dir: Path):
    """Create route visualization as a clear flow diagram."""
    routes = solution['routes']
    checkpoint = int(sheet_name.strip())
    num_routes = len(routes)

    # Calculate figure size based on routes
    fig_height = max(8, num_routes * 6)
    fig, axes = plt.subplots(num_routes, 1, figsize=(20, fig_height))

    if num_routes == 1:
        axes = [axes]

    route_colors = [COLORS['route1'], COLORS['route2'], '#4CAF50', '#FF9800']

    for idx, (ax, route) in enumerate(zip(axes, routes)):
        route_nodes = route['route']
        n = len(route_nodes)
        color = route_colors[idx % len(route_colors)]

        # Calculate grid layout
        nodes_per_row = 15
        num_rows = math.ceil(n / nodes_per_row)

        # Draw route as a snake pattern
        positions = []
        for i in range(n):
            row = i // nodes_per_row
            col = i % nodes_per_row
            # Reverse direction for odd rows (snake pattern)
            if row % 2 == 1:
                col = nodes_per_row - 1 - col
            x = col * 1.2
            y = -row * 1.5
            positions.append((x, y))

        # Draw connections
        for i in range(n - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.6,
                                      connectionstyle='arc3,rad=0'))

        # Draw nodes with labels
        for i, node_id in enumerate(route_nodes):
            x, y = positions[i]

            if node_id == 1:  # Depot
                node_color = COLORS['depot']
                size = 500
                text_color = 'white'
            elif node_id == checkpoint:  # Checkpoint
                node_color = COLORS['checkpoint']
                size = 450
                text_color = 'white'
            else:
                node_color = '#E3F2FD'  # Light blue for collection nodes
                size = 350
                text_color = 'black'

            ax.scatter(x, y, c=node_color, s=size, zorder=5, edgecolors=color, linewidth=2)
            ax.annotate(str(node_id), (x, y), ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color, zorder=6)

        # Set axis limits
        ax.set_xlim(-1, nodes_per_row * 1.2)
        ax.set_ylim(-num_rows * 1.5 - 0.5, 1)
        ax.axis('off')

        # Title for each route
        ax.set_title(f'Vehicle {route["vehicle_id"]} (Type {route["vehicle_type"]})\n'
                    f'Stops: {n-2} collection nodes | Distance: {route["distance_km"]:.2f} km | '
                    f'Load: {route["general_load"]:.0f} general, {route["recycle_load"]:.0f} recycle',
                    fontsize=12, fontweight='bold', loc='left', pad=10)

    # Add legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['depot'], edgecolor='black', label='Depot (Node 1) - Start & End'),
        mpatches.Patch(facecolor=COLORS['checkpoint'], edgecolor='black', label=f'Checkpoint (Node {checkpoint}) - Must visit before return'),
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='black', label='Collection Nodes'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'Route Map - {sheet_name} Zones\n'
                f'Total: {solution["num_vehicles_used"]} Vehicle(s), '
                f'{solution["total_distance_km"]:.2f} km, '
                f'{solution["total_cost"]:,.2f} THB',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_file = output_dir / f'route_{sheet_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_file.name}')


def create_route_sequence_chart(solution: dict, sheet_name: str, output_dir: Path):
    """Create route sequence diagram showing ALL node numbers clearly."""
    routes = solution['routes']
    checkpoint = int(sheet_name.strip())
    num_routes = len(routes)

    route_colors = [COLORS['route1'], COLORS['route2'], '#4CAF50', '#FF9800']

    for route_idx, route in enumerate(routes):
        route_nodes = route['route']
        n = len(route_nodes)
        color = route_colors[route_idx % len(route_colors)]

        # Calculate layout - multiple rows if needed
        nodes_per_row = 20
        num_rows = math.ceil(n / nodes_per_row)

        fig_width = min(24, max(16, nodes_per_row * 1.1))
        fig_height = max(4, num_rows * 2.5 + 2)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Draw route as rows
        for i, node_id in enumerate(route_nodes):
            row = i // nodes_per_row
            col = i % nodes_per_row

            x = col * 1.1
            y = -row * 2

            # Node styling
            if node_id == 1:  # Depot
                node_color = COLORS['depot']
                size = 600
                text_color = 'white'
                fontsize = 11
            elif node_id == checkpoint:  # Checkpoint
                node_color = COLORS['checkpoint']
                size = 550
                text_color = 'white'
                fontsize = 11
            else:
                node_color = '#BBDEFB'  # Light blue
                size = 450
                text_color = 'black'
                fontsize = 9

            # Draw node
            ax.scatter(x, y, c=node_color, s=size, zorder=5, edgecolors=color, linewidth=2)

            # Draw node number - ALWAYS show ALL numbers
            ax.annotate(str(node_id), (x, y), ha='center', va='center',
                       fontsize=fontsize, fontweight='bold', color=text_color, zorder=6)

            # Draw arrow to next node
            if i < n - 1:
                next_row = (i + 1) // nodes_per_row
                next_col = (i + 1) % nodes_per_row
                next_x = next_col * 1.1
                next_y = -next_row * 2

                # Different arrow style for same row vs different row
                if row == next_row:
                    # Same row - straight arrow
                    ax.annotate('', xy=(next_x - 0.35, next_y), xytext=(x + 0.35, y),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
                else:
                    # Different row - curved arrow going down
                    ax.annotate('', xy=(next_x, next_y + 0.5), xytext=(x, y - 0.5),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7,
                                             connectionstyle='arc3,rad=-0.3'))

        # Add step numbers below each node
        for i, node_id in enumerate(route_nodes):
            row = i // nodes_per_row
            col = i % nodes_per_row
            x = col * 1.1
            y = -row * 2

            # Step number (smaller, below node)
            step_label = 'START' if i == 0 else ('END' if i == n-1 else f'#{i}')
            ax.annotate(step_label, (x, y - 0.7), ha='center', va='top',
                       fontsize=7, color='gray', style='italic')

        # Set axis limits
        ax.set_xlim(-1, nodes_per_row * 1.1)
        ax.set_ylim(-num_rows * 2 - 1.5, 1.5)
        ax.axis('off')

        # Title
        ax.set_title(f'Route Sequence - {sheet_name} Zones - Vehicle {route["vehicle_id"]} (Type {route["vehicle_type"]})\n'
                    f'Total Stops: {n} ({n-2} collection + depot + checkpoint) | '
                    f'Distance: {route["distance_km"]:.2f} km | '
                    f'Load: {route["general_load"]:.0f} general, {route["recycle_load"]:.0f} recycle',
                    fontsize=12, fontweight='bold', pad=15)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['depot'], edgecolor=color, linewidth=2,
                          label='Depot (Node 1) - Start & End'),
            mpatches.Patch(facecolor=COLORS['checkpoint'], edgecolor=color, linewidth=2,
                          label=f'Checkpoint (Node {checkpoint})'),
            mpatches.Patch(facecolor='#BBDEFB', edgecolor=color, linewidth=2,
                          label='Collection Node'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Add route text summary at bottom
        route_text = ' -> '.join(map(str, route_nodes))
        if len(route_text) > 200:
            # Split into multiple lines
            parts = []
            current = ""
            for node in route_nodes:
                if current:
                    current += f" -> {node}"
                else:
                    current = str(node)
                if len(current) > 100:
                    parts.append(current)
                    current = ""
            if current:
                parts.append(current)
            route_text = '\n'.join(parts)

        ax.text(0.5, -num_rows * 2 - 1, f'Full Route: {route_text}',
               transform=ax.transData, fontsize=8, ha='center', va='top',
               wrap=True, family='monospace',
               bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))

        plt.tight_layout()

        # Save with vehicle number in filename if multiple vehicles
        if num_routes > 1:
            output_file = output_dir / f'route_sequence_{sheet_name}_v{route["vehicle_id"]}.png'
        else:
            output_file = output_dir / f'route_sequence_{sheet_name}.png'

        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Saved: {output_file.name}')


def create_summary_comparison(solutions: dict, output_dir: Path):
    """Create summary comparison charts for all problems."""
    sheets = sorted(solutions.keys(), key=lambda x: int(x.strip()))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Data preparation
    labels = [f'{s.strip()} Zones' for s in sheets]
    distances = [solutions[s]['total_distance_km'] for s in sheets]
    total_costs = [solutions[s]['total_cost'] for s in sheets]
    fixed_costs = [solutions[s]['total_fixed_cost'] for s in sheets]
    fuel_costs = [solutions[s]['total_fuel_cost'] for s in sheets]
    vehicles = [solutions[s]['num_vehicles_used'] for s in sheets]

    x = np.arange(len(sheets))

    # Chart 1: Distance comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(x, distances, color=COLORS['primary'])
    ax1.set_xlabel('Problem Size', fontsize=12)
    ax1.set_ylabel('Total Distance (km)', fontsize=12)
    ax1.set_title('Total Distance by Problem Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    for bar, val in zip(bars, distances):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Chart 2: Total cost comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(x, total_costs, color=COLORS['accent'])
    ax2.set_xlabel('Problem Size', fontsize=12)
    ax2.set_ylabel('Total Cost (THB)', fontsize=12)
    ax2.set_title('Total Cost by Problem Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    for bar, val in zip(bars, total_costs):
        ax2.annotate(f'{val:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Chart 3: Cost breakdown stacked bar
    ax3 = axes[1, 0]
    width = 0.6
    bars1 = ax3.bar(x, fixed_costs, width, label='Fixed Cost', color=COLORS['fixed_cost'])
    bars2 = ax3.bar(x, fuel_costs, width, bottom=fixed_costs, label='Fuel Cost', color=COLORS['fuel_cost'])
    ax3.set_xlabel('Problem Size', fontsize=12)
    ax3.set_ylabel('Cost (THB)', fontsize=12)
    ax3.set_title('Cost Breakdown by Problem Size', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.legend()

    # Chart 4: Vehicles used
    ax4 = axes[1, 1]
    bars = ax4.bar(x, vehicles, color=COLORS['secondary'])
    ax4.set_xlabel('Problem Size', fontsize=12)
    ax4.set_ylabel('Number of Vehicles', fontsize=12)
    ax4.set_title('Vehicles Used by Problem Size', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_ylim(0, max(vehicles) + 1)
    for bar, val in zip(bars, vehicles):
        ax4.annotate(f'{val}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'summary_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_file.name}')


def create_detailed_summary_table(solutions: dict, output_dir: Path):
    """Create a detailed summary table as an image."""
    sheets = sorted(solutions.keys(), key=lambda x: int(x.strip()))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    headers = ['Problem', 'Nodes', 'Vehicles', 'Distance\n(km)', 'Fixed Cost\n(THB)',
               'Fuel Cost\n(THB)', 'Total Cost\n(THB)', 'Status']

    table_data = []
    for s in sheets:
        sol = solutions[s]
        table_data.append([
            f'{s.strip()} Zones',
            str(int(s.strip())),
            str(sol['num_vehicles_used']),
            f'{sol["total_distance_km"]:.2f}',
            f'{sol["total_fixed_cost"]:,.0f}',
            f'{sol["total_fuel_cost"]:.2f}',
            f'{sol["total_cost"]:,.2f}',
            sol['status']
        ])

    # Add totals row
    total_distance = sum(solutions[s]['total_distance_km'] for s in sheets)
    total_fixed = sum(solutions[s]['total_fixed_cost'] for s in sheets)
    total_fuel = sum(solutions[s]['total_fuel_cost'] for s in sheets)
    total_cost = sum(solutions[s]['total_cost'] for s in sheets)
    total_vehicles = sum(solutions[s]['num_vehicles_used'] for s in sheets)

    table_data.append([
        'TOTAL', '-', str(total_vehicles), f'{total_distance:.2f}',
        f'{total_fixed:,.0f}', f'{total_fuel:.2f}', f'{total_cost:,.2f}', '-'
    ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=[COLORS['primary']] * len(headers))

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style total row
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#E0E0E0')
        table[(len(table_data), i)].set_text_props(fontweight='bold')

    ax.set_title('VRP Solver v2 - Results Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_file = output_dir / 'summary_table.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_file.name}')


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("VRP Solver v2 - Visualization Generator")
    print("=" * 60)

    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results_v2'
    output_dir = base_dir / 'visualizations' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sheet configurations
    sheets = ['20', '30', '50', '80', '138']

    # Load all solutions
    solutions = {}
    for sheet in sheets:
        sheet_key = f'{sheet} ' if sheet == '20' else sheet
        file_name = f'vrp_solution_v2_{sheet}.json'
        file_path = results_dir / file_name

        if file_path.exists():
            solutions[sheet_key] = load_solution(file_path)
            print(f"Loaded: {file_name}")
        else:
            print(f"Warning: {file_name} not found")

    print("\nGenerating visualizations...")

    # Generate individual charts for each problem
    for sheet in sheets:
        sheet_key = f'{sheet} ' if sheet == '20' else sheet
        if sheet_key in solutions:
            print(f"\n{sheet} Zones:")
            solution = solutions[sheet_key]

            # Cost breakdown
            create_cost_breakdown_chart(solution, sheet, output_dir)

            # Capacity utilization
            create_capacity_chart(solution, sheet, output_dir)

            # Route visualization (circular layout)
            create_route_visualization(solution, sheet, output_dir)

            # Route sequence diagram
            create_route_sequence_chart(solution, sheet, output_dir)

    # Generate summary charts
    print("\nSummary Charts:")
    create_summary_comparison(solutions, output_dir)
    create_detailed_summary_table(solutions, output_dir)

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

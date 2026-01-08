# Custom Airplane Projects Feature

## Overview
The financial model now supports adding custom airplane projects with persistent local storage. This allows you to model additional aircraft types (like CJ4, Phenom 300, etc.) without modifying the core application code.

## Features Implemented

### 1. Hull Value Display
- **Average hull values** are now displayed beside the price sliders for all airplane projects:
  - 525 Winglet: $4.5M
  - 510 Mustang: $2.5M
  - Challenger 604: $12.0M
  - CJ4: $9.0M (available for custom projects)

### 2. Add Custom Airplane Projects
You can add new airplane projects through the UI with the following parameters:
- **Project Name**: e.g., "CJ4", "Phenom 300"
- **TAM (Total Addressable Market)**: Number of aircraft in the market
- **Average Hull Value ($M)**: Market value of the aircraft
- **Base Winglet Price ($k)**: Starting price for winglets
- **COGS per Unit ($k)**: Cost of goods sold per unit
- **Certification Start Year**: When certification begins (2026-2030)
- **Certification Duration (Quarters)**: How long certification takes (1-20 quarters)
- **Certification Cost ($M)**: Total certification cost

### 3. Local Storage System
- Custom projects are saved to `custom_projects.json` in the same directory as the app
- This file is automatically created when you save your first custom project
- Projects persist between sessions - they'll be available next time you run the app
- Each user's custom projects are stored locally on their computer
- The file doesn't affect the base application or other users

### 4. Project Management
- **Add Projects**: Use the "➕ Add New Custom Airplane Project" expander
- **View Projects**: Use the "📋 Manage Custom Projects" expander to see all saved projects
- **Delete Projects**: Click the 🗑️ Delete button next to any project
- **Enable/Disable**: Use checkboxes in the Project Selection area

### 5. Full Integration
Custom projects are fully integrated into the financial model:
- Revenue calculations with price escalation
- COGS tracking
- Certification cost spreading over time
- TAM penetration forecasting
- Market penetration metrics
- Included in total revenue, COGS, and enterprise value calculations

## How to Use

### Adding a CJ4 Project (Example)
1. Click "➕ Add New Custom Airplane Project" to expand the form
2. Enter the following:
   - Project Name: `CJ4`
   - TAM: `500` (or your estimate)
   - Average Hull Value: `9.0` ($9M)
   - Base Winglet Price: `300` ($300k)
   - COGS per Unit: `150` ($150k)
   - Certification Start Year: `2026`
   - Certification Duration: `8` quarters
   - Certification Cost: `10.0` ($10M)
3. Click "💾 Save Project"
4. The project will appear in the Project Selection area
5. Check the box to enable it
6. Configure the project parameters in the sidebar (timeline, pricing, forecast)

### Viewing and Managing Projects
1. Click "📋 Manage Custom Projects" to see all saved projects
2. Review project details (TAM, Hull Value, Price)
3. Click 🗑️ Delete to remove any project
4. The project list updates immediately

### How Local Storage Works
- **File Location**: `custom_projects.json` in the app directory
- **Format**: JSON array of project objects
- **Persistence**: Survives app restarts and system reboots
- **User-Specific**: Each user has their own file on their computer
- **No Cloud Sync**: Projects are stored locally only
- **Backup**: You can copy the JSON file to backup your custom projects

### Example JSON Structure
```json
[
  {
    "name": "CJ4",
    "tam": 500,
    "hull_value": 9.0,
    "price_base": 300,
    "cogs": 150,
    "cert_start_year": 2026,
    "cert_duration_qtrs": 8,
    "cert_cost": 10.0
  }
]
```

## Technical Details

### Data Flow
1. **Load**: Custom projects load from JSON file on app startup
2. **Display**: Projects appear in UI with checkboxes
3. **Configure**: Sidebar sliders allow parameter adjustment
4. **Calculate**: Forecasts generated using same logic as standard projects
5. **Integrate**: Revenue/costs flow into main financial model
6. **Save**: New projects saved to JSON file immediately

### Calculation Logic
Custom projects use the same penetration-based forecasting as standard projects:
- TAM-based market sizing
- Linear penetration ramp from start year to 2035
- Annual unit sales calculated to hit penetration targets
- Price escalation applied year-over-year
- Certification costs spread evenly over certification period

## Benefits

1. **No Code Changes**: Add new projects without modifying the application
2. **Persistent**: Projects saved between sessions
3. **User-Specific**: Each user maintains their own project list
4. **Flexible**: All key parameters are configurable
5. **Integrated**: Full financial model integration
6. **Scalable**: Add unlimited custom projects

## Troubleshooting

**Q: My custom projects disappeared**
A: Check if `custom_projects.json` exists in the app directory. It may have been deleted.

**Q: Can I share my custom projects with others?**
A: Yes, copy the `custom_projects.json` file and share it. Others can place it in their app directory.

**Q: How do I backup my projects?**
A: Copy the `custom_projects.json` file to a safe location.

**Q: Can I edit the JSON file directly?**
A: Yes, but make sure to maintain valid JSON format. The app will warn if it can't load the file.

**Q: What happens if I delete a project that's enabled?**
A: The app will remove it from calculations immediately and update the display.

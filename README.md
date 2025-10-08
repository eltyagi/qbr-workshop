# AI Discovery Questions Timer

A minimal, modern web application for tracking customer conversations about AI implementations. Features a timer, progress bar, and comprehensive checklist of discovery questions.

## Features

- **Timer**: Track conversation duration with start, pause, and reset functionality
- **Progress Bar**: Visual progress indicator based on completed questions
- **Interactive Checklist**: 15 essential AI discovery questions for customer conversations
- **Auto-save**: Automatically saves progress to localStorage
- **Keyboard Shortcuts**: 
  - `Space`: Start/Pause timer
  - `R`: Reset everything
- **Responsive Design**: Works on desktop and mobile devices

## Usage

1. Open `index.html` in a web browser
2. Click "Start" to begin timing your conversation
3. Check off questions as you cover them with the customer
4. Progress bar updates automatically
5. View summary when all questions are completed

## Discovery Questions Included

The application includes 15 key discovery questions covering:
- Business challenges and objectives
- Current processes and pain points  
- Technical requirements and constraints
- Budget and timeline considerations
- Success metrics and expectations
- Integration and compliance needs

## Running Locally

### Option 1: Direct File Access
Simply open `index.html` in any modern web browser.

### Option 2: Local Server (Recommended)
```bash
# Using Python 3
python3 -m http.server 8000

# Using Node.js (if you have it installed)
npx serve .

# Then open http://localhost:8000 in your browser
```

## Files Structure

```
├── index.html      # Main HTML structure
├── style.css       # Modern, minimal styling
├── script.js       # Timer logic and interactivity
└── README.md       # This file
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Customization

You can easily customize the discovery questions by editing the `discoveryQuestions` array in `script.js`.

---

**Note**: This tool is designed to enhance customer conversations about AI, not replace human judgment and expertise in understanding customer needs.
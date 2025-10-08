// AI Discovery Questions Data
const discoveryQuestions = [
    "What specific business challenges are you hoping AI could help solve?",
    "How are you currently handling the processes that AI might improve?",
    "What's your current level of experience with AI technologies?",
    "Who would be the primary users of an AI solution in your organization?",
    "What data sources do you have available that AI could leverage?",
    "How do you currently measure success in the areas AI might impact?",
    "What concerns do you have about implementing AI in your business?",
    "What's your timeline for exploring or implementing AI solutions?",
    "What budget range are you considering for AI initiatives?",
    "How important is it that the AI solution integrates with your existing systems?",
    "What level of AI explainability or transparency do you need?",
    "How do you envision training your team on new AI tools?",
    "What compliance or regulatory requirements must any AI solution meet?",
    "Have you evaluated any AI solutions before? What was that experience like?",
    "What would a successful AI implementation look like for your organization?"
];

// Application State
let timerState = {
    isRunning: false,
    startTime: null,
    elapsedTime: 0,
    interval: null
};

let questionsState = {
    questions: discoveryQuestions.map((text, index) => ({
        id: index,
        text: text,
        completed: false
    })),
    completedCount: 0
};

// DOM Elements
const timerDisplay = document.getElementById('timer');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const progressFill = document.getElementById('progressFill');
const progressCounter = document.getElementById('progressCounter');
const questionsContainer = document.getElementById('questionsContainer');
const summarySection = document.getElementById('summarySection');
const finalTime = document.getElementById('finalTime');
const finalCount = document.getElementById('finalCount');

// Timer Functions
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateTimerDisplay() {
    timerDisplay.textContent = formatTime(Math.floor(timerState.elapsedTime / 1000));
}

function startTimer() {
    if (!timerState.isRunning) {
        timerState.isRunning = true;
        timerState.startTime = Date.now() - timerState.elapsedTime;
        
        timerState.interval = setInterval(() => {
            timerState.elapsedTime = Date.now() - timerState.startTime;
            updateTimerDisplay();
        }, 1000);
        
        startBtn.disabled = true;
        pauseBtn.disabled = false;
    }
}

function pauseTimer() {
    if (timerState.isRunning) {
        timerState.isRunning = false;
        clearInterval(timerState.interval);
        
        startBtn.disabled = false;
        pauseBtn.disabled = true;
    }
}

function resetTimer() {
    timerState.isRunning = false;
    timerState.elapsedTime = 0;
    clearInterval(timerState.interval);
    
    updateTimerDisplay();
    startBtn.disabled = false;
    pauseBtn.disabled = true;
}

// Progress Functions
function updateProgress() {
    const totalQuestions = questionsState.questions.length;
    const completedQuestions = questionsState.completedCount;
    const percentage = totalQuestions > 0 ? (completedQuestions / totalQuestions) * 100 : 0;
    
    progressFill.style.width = `${percentage}%`;
    progressCounter.textContent = `${completedQuestions}/${totalQuestions}`;
    
    // Show summary if all questions are completed
    if (completedQuestions === totalQuestions && totalQuestions > 0) {
        showSummary();
    }
}

// Questions Functions
function createQuestionElement(question) {
    const questionItem = document.createElement('div');
    questionItem.className = 'question-item';
    questionItem.dataset.id = question.id;
    
    const checkbox = document.createElement('div');
    checkbox.className = `question-checkbox ${question.completed ? 'checked' : ''}`;
    
    const text = document.createElement('div');
    text.className = 'question-text';
    text.textContent = question.text;
    
    questionItem.appendChild(checkbox);
    questionItem.appendChild(text);
    
    if (question.completed) {
        questionItem.classList.add('completed');
    }
    
    // Add click handler
    questionItem.addEventListener('click', () => toggleQuestion(question.id));
    
    return questionItem;
}

function toggleQuestion(questionId) {
    const question = questionsState.questions.find(q => q.id === questionId);
    if (question) {
        question.completed = !question.completed;
        
        // Update completed count
        questionsState.completedCount = questionsState.questions.filter(q => q.completed).length;
        
        // Re-render questions
        renderQuestions();
        updateProgress();
    }
}

function renderQuestions() {
    questionsContainer.innerHTML = '';
    
    questionsState.questions.forEach(question => {
        const questionElement = createQuestionElement(question);
        questionsContainer.appendChild(questionElement);
    });
}

// Summary Functions
function showSummary() {
    pauseTimer();
    finalTime.textContent = formatTime(Math.floor(timerState.elapsedTime / 1000));
    finalCount.textContent = progressCounter.textContent;
    summarySection.style.display = 'block';
    summarySection.scrollIntoView({ behavior: 'smooth' });
}

function resetAll() {
    // Reset timer
    resetTimer();
    
    // Reset questions
    questionsState.questions.forEach(question => {
        question.completed = false;
    });
    questionsState.completedCount = 0;
    
    // Re-render
    renderQuestions();
    updateProgress();
    
    // Hide summary
    summarySection.style.display = 'none';
}

// Event Listeners
startBtn.addEventListener('click', startTimer);
pauseBtn.addEventListener('click', pauseTimer);
resetBtn.addEventListener('click', resetAll);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) return; // Ignore shortcuts when Ctrl/Cmd is pressed
    
    switch(e.code) {
        case 'Space':
            e.preventDefault();
            if (timerState.isRunning) {
                pauseTimer();
            } else {
                startTimer();
            }
            break;
        case 'KeyR':
            e.preventDefault();
            resetAll();
            break;
    }
});

// Initialize Application
function initializeApp() {
    // Set initial button states
    pauseBtn.disabled = true;
    
    // Render questions
    renderQuestions();
    updateProgress();
    
    // Set initial timer display
    updateTimerDisplay();
    
    console.log('AI Discovery Questions Timer initialized');
    console.log('Keyboard shortcuts:');
    console.log('- Space: Start/Pause timer');
    console.log('- R: Reset everything');
}

// Auto-save functionality (optional - saves state to localStorage)
function saveState() {
    const state = {
        timer: {
            elapsedTime: timerState.elapsedTime,
            isRunning: false // Never save as running
        },
        questions: questionsState.questions.map(q => ({
            id: q.id,
            completed: q.completed
        }))
    };
    localStorage.setItem('aiDiscoveryState', JSON.stringify(state));
}

function loadState() {
    const saved = localStorage.getItem('aiDiscoveryState');
    if (saved) {
        try {
            const state = JSON.parse(saved);
            
            // Restore timer (but not running state)
            if (state.timer) {
                timerState.elapsedTime = state.timer.elapsedTime || 0;
            }
            
            // Restore questions
            if (state.questions && Array.isArray(state.questions)) {
                state.questions.forEach(savedQ => {
                    const question = questionsState.questions.find(q => q.id === savedQ.id);
                    if (question) {
                        question.completed = savedQ.completed;
                    }
                });
                questionsState.completedCount = questionsState.questions.filter(q => q.completed).length;
            }
        } catch (e) {
            console.warn('Could not restore saved state:', e);
        }
    }
}

// Auto-save on changes
function setupAutoSave() {
    // Save state when questions change
    const originalToggleQuestion = toggleQuestion;
    window.toggleQuestion = function(questionId) {
        originalToggleQuestion(questionId);
        saveState();
    };
    
    // Save timer state periodically
    setInterval(() => {
        if (timerState.isRunning || timerState.elapsedTime > 0) {
            saveState();
        }
    }, 5000);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    loadState();
    initializeApp();
    setupAutoSave();
});

// Handle page visibility change (pause timer when tab is not visible)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && timerState.isRunning) {
        // Timer continues running in background, which is the desired behavior
        // for tracking total conversation time
    }
});
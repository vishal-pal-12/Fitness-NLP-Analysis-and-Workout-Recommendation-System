import pandas as pd
import json
import os

# Base directory
BASE_DIR = r"C:\Users\VISHAL PAL\.gemini\antigravity\scratch\fitness_nlp_project"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. GENERATE WORKOUT PLANS DATASET (50+ diverse workouts)
# -------------------------------------------------------------
workouts = [
    {
        "workout_id": 1,
        "title": "Full Body HIIT Blast",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 30,
        "target_muscles": "Full Body, Quads, Core, Shoulders",
        "equipment_needed": "Bodyweight / None",
        "calories_burned_est": 350,
        "description": "High intensity interval training routine designed to maximize calorie burn and metabolic rate without any weights. Features explosive plyometrics, mountain climbers, and burpees.",
        "exercises": "Jumping Jacks; Burpees; Mountain Climbers; High Knees; Bodyweight Squats; Plank to Push-up",
        "tags": "weight loss; hiit; cardio; bodyweight; fat burn; stamina; home workout; no equipment"
    },
    {
        "workout_id": 2,
        "title": "Hypertrophy Push Day: Chest, Shoulders & Triceps",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 60,
        "target_muscles": "Chest, Triceps, Anterior Deltoids",
        "equipment_needed": "Dumbbells, Barbell, Bench",
        "calories_burned_est": 420,
        "description": "Comprehensive upper body pushing session focused on muscle hypertrophy and progressive overload. Targets pectoral development, shoulder strength, and tricep definition.",
        "exercises": "Barbell Bench Press; Incline Dumbbell Press; Overhead Barbell Press; Dumbbell Lateral Raises; Tricep Rope Pushdowns; Dips",
        "tags": "muscle building; hypertrophy; chest; shoulders; triceps; upper body; barbell; dumbbells; strength"
    },
    {
        "workout_id": 3,
        "title": "PPL Pull Day: Back, Rear Delts & Biceps",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 60,
        "target_muscles": "Back, Latissimus Dorsi, Biceps, Traps",
        "equipment_needed": "Barbell, Dumbbells, Pull-up Bar",
        "calories_burned_est": 410,
        "description": "Targeted pulling workout focusing on back width, upper back thickness, and bicep peak building. Incorporates vertical and horizontal pulling movements.",
        "exercises": "Deadlifts; Weighted Pull-ups; Bent-Over Barbell Rows; Seated Cable Rows; Barbell Bicep Curls; Incline Dumbbell Hammer Curls",
        "tags": "back; biceps; pull; hypertrophy; muscle building; lats; barbell; pulling; upper body"
    },
    {
        "workout_id": 4,
        "title": "Lower Body Quad & Glute Destroyer",
        "category": "Strength",
        "difficulty_level": "Advanced",
        "duration_minutes": 65,
        "target_muscles": "Quads, Glutes, Hamstrings, Calves",
        "equipment_needed": "Barbell, Squat Rack, Dumbbells",
        "calories_burned_est": 520,
        "description": "Heavy leg day routine designed to build lower body strength and muscular power. Features barbell squats, Romanian deadlifts, and walking lunges for complete quad and glute activation.",
        "exercises": "Barbell Back Squats; Romanian Deadlifts; Bulgarian Split Squats; Leg Press; Standing Calf Raises; Hamstring Leg Curls",
        "tags": "legs; lower body; quads; glutes; hamstrings; squat; strength; heavy; barbell"
    },
    {
        "workout_id": 5,
        "title": "Beginner Calisthenics & Bodyweight Mastery",
        "category": "Calisthenics",
        "difficulty_level": "Beginner",
        "duration_minutes": 35,
        "target_muscles": "Full Body, Chest, Back, Core",
        "equipment_needed": "Pull-up Bar, Bodyweight / None",
        "calories_burned_est": 260,
        "description": "Foundational calisthenics routine for beginners seeking upper body control and bodyweight strength. Teaches clean form for pushups, inverted rows, and core stabilization.",
        "exercises": "Knee Push-ups; Regular Push-ups; Inverted Rows; Bodyweight Squats; Dead Hang; Hanging Knee Raises",
        "tags": "calisthenics; beginner; bodyweight; functional; core; push-ups; pull-up bar; home workout"
    },
    {
        "workout_id": 6,
        "title": "Vinyasa Flow Mobility & Deep Flexibility",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 45,
        "target_muscles": "Hamstrings, Hip Flexors, Spine, Shoulders",
        "equipment_needed": "Yoga Mat",
        "calories_burned_est": 180,
        "description": "Soothing and rejuvenating yoga session to enhance joint mobility, lengthen tight hamstrings, open tight hips, and alleviate lower back stiffness after intense lifting.",
        "exercises": "Sun Salutations; Downward Facing Dog; Warrior II; Pigeon Pose; Cobra Pose; Seated Forward Bend; Child's Pose",
        "tags": "yoga; mobility; flexibility; recovery; stretching; low impact; mindfulness; beginner; hip flexors"
    },
    {
        "workout_id": 7,
        "title": "Starting Strength 5x5 Power Routine",
        "category": "Powerlifting",
        "difficulty_level": "Intermediate",
        "duration_minutes": 55,
        "target_muscles": "Quads, Chest, Back, Posterior Chain",
        "equipment_needed": "Barbell, Squat Rack, Bench",
        "calories_burned_est": 390,
        "description": "Classic 5x5 compound strength training program focusing on multi-joint functional lifts. Ideal for building raw strength, bone density, and core stability.",
        "exercises": "Low Bar Squat 5x5; Barbell Overhead Press 5x5; Conventional Deadlift 1x5; Barbell Row 5x5",
        "tags": "powerlifting; 5x5; strength; compound lifts; squat; bench press; deadlift; barbell; mass"
    },
    {
        "workout_id": 8,
        "title": "Core Sculpt & Six-Pack Shredder",
        "category": "Core & Abs",
        "difficulty_level": "Intermediate",
        "duration_minutes": 25,
        "target_muscles": "Rectus Abdominis, Obliques, Transverse Abdominis",
        "equipment_needed": "Bodyweight / None, Exercise Mat",
        "calories_burned_est": 210,
        "description": "Focused abdominal circuit targeting upper abs, lower abs, and obliques with isometric holds and rotational power movements to build a rock-solid core.",
        "exercises": "Hanging Leg Raises; Bicycle Crunches; Russian Twists; Ab Rollouts; Side Planks; Hollow Body Hold",
        "tags": "abs; core; six pack; obliques; abdominal; stomach; waist; bodyweight; mat"
    },
    {
        "workout_id": 9,
        "title": "Low-Impact Cardio for Seniors & Beginners",
        "category": "Cardio / HIIT",
        "difficulty_level": "Beginner",
        "duration_minutes": 30,
        "target_muscles": "Legs, Cardiovascular System, Joints",
        "equipment_needed": "Bodyweight / None",
        "calories_burned_est": 190,
        "description": "Gentle cardio session without running or jumping, ideal for seniors, joint pain sufferers, or individuals recovering from lower body injuries. Protects knees and spine.",
        "exercises": "Marching in Place; Step Jacks; Gentle Knee Lifts; Wall Push-ups; Seated Leg Extensions; Standing Calf Raises",
        "tags": "low impact; senior fitness; beginner; joint friendly; gentle; cardio; walking; recovery; knees"
    },
    {
        "workout_id": 10,
        "title": "Kettlebell Metabolic Circuit Shredder",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 40,
        "target_muscles": "Glutes, Hamstrings, Shoulders, Core",
        "equipment_needed": "Kettlebell",
        "calories_burned_est": 450,
        "description": "Dynamic full-body kettlebell circuit combining explosive hip hinges with continuous cardio demands. Supercharges fat burning and muscular endurance.",
        "exercises": "Kettlebell Swings; Goblet Squats; Turkish Get-Ups; Single-Arm Kettlebell Snatches; Kettlebell Clean and Press; Farmer's Walk",
        "tags": "kettlebell; metabolic; fat burn; endurance; full body; posterior chain; functional; kettlebell swings"
    },
    {
        "workout_id": 11,
        "title": "Upper Body Dumbbell Sculpt & Tone",
        "category": "Hypertrophy",
        "difficulty_level": "Beginner",
        "duration_minutes": 40,
        "target_muscles": "Chest, Back, Biceps, Triceps, Shoulders",
        "equipment_needed": "Dumbbells, Bench",
        "calories_burned_est": 290,
        "description": "Dumbbell-only upper body routine perfect for home gyms and beginners wanting to tone their arms, firm their shoulders, and strengthen posture muscles.",
        "exercises": "Dumbbell Chest Press; Bent-Over Dumbbell Rows; Dumbbell Shoulder Press; Alternating Bicep Curls; Overhead Dumbbell Tricep Extension; Lateral Raises",
        "tags": "dumbbells; upper body; tone; arms; shoulders; chest; back; home workout; beginner"
    },
    {
        "workout_id": 12,
        "title": "Runner's Stamina & Aerobic Base Builder",
        "category": "Endurance",
        "difficulty_level": "Intermediate",
        "duration_minutes": 50,
        "target_muscles": "Cardiovascular, Calves, Quads, Hamstrings",
        "equipment_needed": "Running Shoes / Treadmill",
        "calories_burned_est": 560,
        "description": "Interval and steady-state cardiovascular conditioning program to expand VO2 max, running economy, and aerobic threshold for long distance runners.",
        "exercises": "5-min Dynamic Warmup; 25-min Zone 2 Aerobic Jog; 10-min 400m Track Intervals; 5-min Cool-down Walk; Static Leg Stretches",
        "tags": "running; endurance; stamina; cardio; aerobic; treadmill; legs; vo2 max; track"
    },
    {
        "workout_id": 13,
        "title": "Desk Worker Posture & Spinal Rehab",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 25,
        "target_muscles": "Upper Back, Neck, Glutes, Hip Flexors",
        "equipment_needed": "Resistance Bands, Foam Roller",
        "calories_burned_est": 130,
        "description": "Corrective therapeutic exercise routine targeting forward head posture, rounded shoulders, and tight hip flexors caused by prolonged desk sitting.",
        "exercises": "Band Pull-Aparts; Wall Angels; Thoracic Spine Foam Rolling; Glute Bridges; Bird Dog; Cat-Cow Stretch; Chin Tucks",
        "tags": "posture; rehab; spine; neck; back pain; mobility; desk worker; physical therapy; bands"
    },
    {
        "workout_id": 14,
        "title": "Chest Peak & Pectoral Mass Specialist",
        "category": "Hypertrophy",
        "difficulty_level": "Advanced",
        "duration_minutes": 55,
        "target_muscles": "Pectoralis Major, Pectoralis Minor, Front Delts",
        "equipment_needed": "Barbell, Dumbbells, Cable Machine, Bench",
        "calories_burned_est": 380,
        "description": "High-volume chest workout specifically isolating upper, mid, and lower pectoral fibers with varied angles, drop sets, and peak contraction pauses.",
        "exercises": "Incline Barbell Bench Press; Flat Dumbbell Flyes; Low-to-High Cable Crossovers; Decline Dumbbell Press; Weighted Chest Dips; Push-ups to Failure",
        "tags": "chest; pec; mass; hypertrophy; bench press; dumbbells; cables; upper body; advanced"
    },
    {
        "workout_id": 15,
        "title": "Tabata Sprint Extreme Fat Burner",
        "category": "Cardio / HIIT",
        "difficulty_level": "Advanced",
        "duration_minutes": 20,
        "target_muscles": "Full Body, Cardiovascular, Legs",
        "equipment_needed": "Bodyweight / None",
        "calories_burned_est": 320,
        "description": "Brutal 20-second work / 10-second rest Tabata protocol pushing heart rates to maximal zone. Maximum calorie afterburn in minimum total time.",
        "exercises": "Sprint in Place; Burpee Tuck Jumps; Mountain Climber Sprints; Jump Squats; Speed Skaters; Plyo Lunges",
        "tags": "tabata; fat burn; extreme; cardio; hiit; sprints; plyometrics; weight loss; short workout"
    },
    {
        "workout_id": 16,
        "title": "Arm Farm: Biceps & Triceps Super-Sets",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Biceps Brachii, Triceps Brachii, Forearms",
        "equipment_needed": "Barbell, Dumbbells, Cable Machine",
        "calories_burned_est": 310,
        "description": "Antagonistic super-set routine alternating between bicep flexors and tricep extensors to drive skin-splitting blood volume and arm circumference growth.",
        "exercises": "EZ-Bar Curl super-set with Skull Crushers; Hammer Curls super-set with Cable Rope Pushdowns; Preacher Curls super-set with Overhead Tricep Extension",
        "tags": "arms; biceps; triceps; curls; supersets; arms day; hypertrophy; dumbbells; barbell"
    },
    {
        "workout_id": 17,
        "title": "Legs & Posterior Chain Power: Deadlift Focus",
        "category": "Strength",
        "difficulty_level": "Advanced",
        "duration_minutes": 60,
        "target_muscles": "Hamstrings, Glutes, Erector Spinae, Lats",
        "equipment_needed": "Barbell, Weight Plates, Chalk",
        "calories_burned_est": 490,
        "description": "Heavy deadlift and posterior chain power routine. Builds explosive hip hinge drive, unbreakable grip, and thick lower back muscularity.",
        "exercises": "Conventional Deadlifts 5x3; Deficit Deadlifts 3x5; Barbell Hip Thrusts; Glute-Ham Raises; Barbell Shrugs; Farmer's Walk",
        "tags": "deadlift; posterior chain; back; hamstrings; glutes; powerlifting; grip; strength; heavy"
    },
    {
        "workout_id": 18,
        "title": "Morning Sunrise Yoga & Core Awakening",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 20,
        "target_muscles": "Core, Spine, Shoulders, Calves",
        "equipment_needed": "Yoga Mat",
        "calories_burned_est": 110,
        "description": "Short, invigorating morning yoga flow designed to shake off sleep inertia, activate core stabilization muscles, and improve day-long energy and alertness.",
        "exercises": "Gentle Neck Rolls; Cat-Cow; Downward Dog; Plank Pose; Upward Facing Dog; Warrior I; Tree Pose",
        "tags": "yoga; morning; wake up; core; stretching; light; beginner; mat; mindfulness"
    },
    {
        "workout_id": 19,
        "title": "Dumbbell Only Home Fat Loss Blitz",
        "category": "Cardio / HIIT",
        "difficulty_level": "Beginner",
        "duration_minutes": 35,
        "target_muscles": "Full Body, Quads, Shoulders, Core",
        "equipment_needed": "Dumbbells",
        "calories_burned_est": 360,
        "description": "Full body dumbbell complex designed for apartment workouts. Rapid sequence of compound movements to elevate metabolic burn while maintaining muscle mass.",
        "exercises": "Dumbbell Thrusters; Dumbbell Renegade Rows; Dumbbell Romanian Deadlifts; Dumbbell Clean and Press; Dumbbell Suitcase Carries",
        "tags": "dumbbells; fat loss; weight loss; home workout; complex; full body; intermediate; cardio"
    },
    {
        "workout_id": 20,
        "title": "Gymnastic Rings Upper Body Strength",
        "category": "Calisthenics",
        "difficulty_level": "Advanced",
        "duration_minutes": 55,
        "target_muscles": "Chest, Lats, Shoulders, Biceps, Core",
        "equipment_needed": "Gymnastic Rings",
        "calories_burned_est": 390,
        "description": "Unstable gymnastic ring workout requiring immense neuromuscular recruitment and core stability. Develops elite upper body tendon strength and muscular control.",
        "exercises": "Ring Muscle-ups; Ring Dips; Ring Push-ups with Turnout; False Grip Pull-ups; Ring Inverted Rows; Ring Rollouts",
        "tags": "rings; calisthenics; bodyweight; advanced; gymnastics; upper body; core; tendon strength"
    },
    {
        "workout_id": 21,
        "title": "Glute Hypertrophy & Hip Thrust Mastery",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 50,
        "target_muscles": "Gluteus Maximus, Gluteus Medius, Hamstrings",
        "equipment_needed": "Barbell, Resistance Bands, Bench",
        "calories_burned_est": 370,
        "description": "Specialized lower body program focusing entirely on glute activation, shape, and strength. Emphasizes heavy hip thrusts and high-rep band burnouts.",
        "exercises": "Barbell Hip Thrusts; Cable Glute Kickbacks; Bulgarian Split Squats; Banded Abductions; Romanian Deadlifts; Frog Pumps",
        "tags": "glutes; booty; hips; lower body; hypertrophy; hip thrust; resistance bands; dumbbells"
    },
    {
        "workout_id": 22,
        "title": "Shoulder Boulder Cap: 3D Deltoid Builder",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Anterior, Lateral, and Posterior Deltoids, Traps",
        "equipment_needed": "Dumbbells, Cables, Barbell",
        "calories_burned_est": 320,
        "description": "Comprehensive shoulder building session targeting all three deltoid heads for wide, capped shoulders and improved overhead strength.",
        "exercises": "Standing Overhead Press; Dumbbell Lateral Raises; Face Pulls; Cable Rear Delt Flyes; Arnold Press; Dumbbell Shrugs",
        "tags": "shoulders; delts; lateral raise; upper body; hypertrophy; overhead press; dumbells"
    },
    {
        "workout_id": 23,
        "title": "Sprint Interval Training (SIT) for Track Athletes",
        "category": "Endurance",
        "difficulty_level": "Advanced",
        "duration_minutes": 35,
        "target_muscles": "Hamstrings, Quads, Calves, Core",
        "equipment_needed": "Track / Running Shoes",
        "calories_burned_est": 480,
        "description": "High velocity sprint repeat workout designed to increase maximum running speed, lactic threshold, and anaerobic power output.",
        "exercises": "Dynamic Hurdle Drills; 60m Sprint Starts x 5; 100m Maximal Sprints x 4; 200m Flying Sprints x 3; Cool-down Walking",
        "tags": "sprints; track; athletics; speed; anaerobic; running; power; legs; endurance"
    },
    {
        "workout_id": 24,
        "title": "Total Body Resistance Band Home Circuit",
        "category": "Strength",
        "difficulty_level": "Beginner",
        "duration_minutes": 30,
        "target_muscles": "Full Body, Chest, Back, Legs, Arms",
        "equipment_needed": "Resistance Bands",
        "calories_burned_est": 230,
        "description": "Joint-friendly portable resistance band workout for travelers and home lifters. Provides continuous muscular tension throughout each range of motion.",
        "exercises": "Banded Squats; Banded Chest Press; Banded Seated Rows; Banded Overhead Press; Banded Bicep Curls; Banded Tricep Pushdowns",
        "tags": "resistance bands; home workout; travel; joint friendly; light; beginner; full body"
    },
    {
        "workout_id": 25,
        "title": "Olympic Weightlifting: Clean and Jerk Focus",
        "category": "Powerlifting",
        "difficulty_level": "Advanced",
        "duration_minutes": 70,
        "target_muscles": "Full Body, Quads, Traps, Shoulders, Back",
        "equipment_needed": "Olympic Barbell, Bumper Plates, Lifting Platform",
        "calories_burned_est": 510,
        "description": "Technical Olympic lifting training focusing on triple extension velocity, front rack catch, and split jerk overhead stability.",
        "exercises": "Clean Pulls; Hang Cleans; Clean and Jerk 5x2; Front Squats 4x3; Push Presses; Overhead Squats",
        "tags": "olympic lifting; clean and jerk; explosive; barbell; advanced; power; weightlifting"
    },
    {
        "workout_id": 26,
        "title": "Back Width Lat Spreader Session",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 50,
        "target_muscles": "Latissimus Dorsi, Teres Major, Lower Traps",
        "equipment_needed": "Lat Pulldown Machine, Cable Machine, Dumbbells",
        "calories_burned_est": 360,
        "description": "Specialized back session aiming to create a classic V-taper physique. Combines wide-grip pulldowns, single-arm pulldowns, and straight-arm cable pushdowns.",
        "exercises": "Wide-Grip Lat Pulldowns; Single-Arm Kneeling Cable Lat Pulldowns; Straight-Arm Pulldowns; Neutral-Grip Pull-ups; Dumbbell Pullovers",
        "tags": "back; lats; v-taper; width; lat pulldown; pull; hypertrophy; upper body"
    },
    {
        "workout_id": 27,
        "title": "Pilates Mat Core & Pelvic Stability",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 35,
        "target_muscles": "Transverse Abdominis, Pelvic Floor, Lower Back",
        "equipment_needed": "Exercise Mat",
        "calories_burned_est": 160,
        "description": "Classical Pilates mat routine emphasizing breath control, pelvic alignment, and deep intrinsic spinal core stabilization.",
        "exercises": "The Hundred; Roll-Up; Single Leg Stretch; Criss-Cross; Swan Dive; Teaser Prep; Pelvic Curls",
        "tags": "pilates; core; pelvic floor; posture; mat; low impact; beginner; flexibility"
    },
    {
        "workout_id": 28,
        "title": "Boxing Heavy Bag Conditioning & Agility",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Shoulders, Core, Calves, Cardiovascular",
        "equipment_needed": "Heavy Bag, Boxing Gloves, Jump Rope",
        "calories_burned_est": 520,
        "description": "High-octane boxing rounds alternating 3-minute striking flurries with jump rope footwork drills for peak cardiovascular fitness and stress relief.",
        "exercises": "Jump Rope 3 Rounds; Jab-Cross Combinations; Hook and Uppercut Drills; Heavy Bag Power Rounds; Shadowboxing with Slip Drills; Plank Burnout",
        "tags": "boxing; cardio; heavy bag; stamina; endurance; agility; shoulders; fat loss; martial arts"
    },
    {
        "workout_id": 29,
        "title": "Squat Specialization & Lower Body Power",
        "category": "Strength",
        "difficulty_level": "Advanced",
        "duration_minutes": 60,
        "target_muscles": "Quads, Adductors, Glutes, Core",
        "equipment_needed": "Barbell, Squat Rack, Weight Belt",
        "calories_burned_est": 460,
        "description": "Intense barbell squat peaking program designed to bust through squat plateaus using pause squats, pin squats, and accessory quad overload.",
        "exercises": "Competition Back Squats 4x4; 2-Second Paused Squats 3x3; Barbell Front Squats 3x5; Walking Dumbbell Lunges; Leg Extensions",
        "tags": "squat; quads; legs; powerlifting; lower body; strength; barbell; heavy lifting"
    },
    {
        "workout_id": 30,
        "title": "Rest Day Active Recovery & Foam Rolling",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 20,
        "target_muscles": "Full Body, Myofascial Tissue, Joints",
        "equipment_needed": "Foam Roller, Lacrosse Ball",
        "calories_burned_est": 90,
        "description": "Gentle active recovery protocol designed to accelerate muscle repair, reduce DOMS (delayed onset muscle soreness), and stimulate lymphatic drainage.",
        "exercises": "IT Band Foam Rolling; Quad and Hamstring Rolling; Thoracic Extension Rolls; Glute Lacrosse Ball Release; Deep Diaphragmatic Breathing",
        "tags": "recovery; rest day; foam rolling; soreness; mobility; active recovery; gentle; joints"
    },
    {
        "workout_id": 31,
        "title": "Swimming Aerobic Engine & Total Body Flow",
        "category": "Endurance",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Full Body, Lats, Shoulders, Cardiovascular",
        "equipment_needed": "Swimming Pool, Goggles",
        "calories_burned_est": 480,
        "description": "Zero-impact cardiovascular conditioning in the pool. Builds lung capacity, shoulder endurance, and full body coordination without joint impact.",
        "exercises": "200m Freestyle Warmup; 4x100m Interval Sprints; 200m Backstroke; 4x50m Kickboard Drills; 100m Cool-down Breaststroke",
        "tags": "swimming; pool; cardio; endurance; zero impact; joints; full body; aerobic; stamina"
    },
    {
        "workout_id": 32,
        "title": "Agility Ladder & Footwork Speed Blitz",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 30,
        "target_muscles": "Calves, Ankles, Quads, Fast-Twitch Fibers",
        "equipment_needed": "Agility Ladder, Cones",
        "calories_burned_est": 310,
        "description": "Dynamic footwork and agility drill session to sharpen neuromuscular coordination, lateral quickness, and multi-directional speed for sports.",
        "exercises": "Ickey Shuffle; In-and-Out Footwork; Lateral High Knees; 5-10-5 Pro Agility Shuttle; Carioca Drill; Box Jump Drills",
        "tags": "agility; footwork; speed; quickness; athletic; ladder; cardio; sports; coordination"
    },
    {
        "workout_id": 33,
        "title": "Barbell Bench Press Plateau Breaker",
        "category": "Strength",
        "difficulty_level": "Advanced",
        "duration_minutes": 55,
        "target_muscles": "Chest, Triceps, Anterior Delts, Upper Back",
        "equipment_needed": "Barbell, Bench Press Station, Resistance Bands",
        "calories_burned_est": 380,
        "description": "Specialized bench press strength cycle targeting lockout velocity and chest drive through paused reps, pin presses, and heavy tricep work.",
        "exercises": "Competition Paused Bench 5x3; Spoto Press 3x4; Pin Press from Chest 3x3; Close-Grip Bench Press; Heavy Dumbbell Rows",
        "tags": "bench press; chest; strength; powerlifting; upper body; heavy; barbell; triceps"
    },
    {
        "workout_id": 34,
        "title": "Gentle Prenatal / Postnatal Pelvic Floor Conditioning",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 25,
        "target_muscles": "Pelvic Floor, Transverse Abdominis, Glutes",
        "equipment_needed": "Exercise Ball, Mat",
        "calories_burned_est": 110,
        "description": "Safe and supportive movement routine focused on deep core re-education, pelvic floor rehabilitation, and gentle hip mobility for mothers.",
        "exercises": "Diaphragmatic Breathing on Ball; Pelvic Tilts; Gentle Glute Bridges; Side-Lying Clamshells; Modified Bird-Dog; Supported Child's Pose",
        "tags": "prenatal; postnatal; pelvic floor; core; gentle; rehab; women's health; beginner"
    },
    {
        "workout_id": 35,
        "title": "Farmer's Walk & Grip Strength Strongman Blitz",
        "category": "Strength",
        "difficulty_level": "Intermediate",
        "duration_minutes": 40,
        "target_muscles": "Forearms, Traps, Core, Upper Back",
        "equipment_needed": "Heavy Dumbbells or Kettlebells, Trap Bar",
        "calories_burned_est": 360,
        "description": "Loaded carry session designed to forge an iron grip, massive trapezoid muscles, and indestructible upright core stability under load.",
        "exercises": "Heavy Farmer's Walk 5x50m; Trap Bar Deadlift Holds; Plate Pinches 4x45s; Suitcase Single-Arm Carries; Barbell Wrist Curls",
        "tags": "grip; forearms; traps; loaded carries; strongman; functional; heavy; core"
    },
    {
        "workout_id": 36,
        "title": "Plyometric Vertical Jump Explosiveness",
        "category": "Powerlifting",
        "difficulty_level": "Intermediate",
        "duration_minutes": 40,
        "target_muscles": "Quads, Calves, Glutes, Hip Extensors",
        "equipment_needed": "Plyo Box, Medicine Ball",
        "calories_burned_est": 340,
        "description": "Explosive stretch-shortening cycle training designed for basketball players and track athletes to maximize vertical jump height and ground force production.",
        "exercises": "Depth Jumps; Seated Box Jumps; Broad Jumps; Medicine Ball Slams; Single-Leg Bound Drills; Continuous Tuck Jumps",
        "tags": "plyometrics; vertical jump; basketball; explosive; power; legs; jumping; athletic"
    },
    {
        "workout_id": 37,
        "title": "Low Back Pain Relief & Lumbar Decompression",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 20,
        "target_muscles": "Lower Back, Erector Spinae, Glutes, Hamstrings",
        "equipment_needed": "Yoga Mat",
        "calories_burned_est": 85,
        "description": "Evidence-based McGill Big 3 spinal hygiene routine that decompresses the lumbar spine and activates deep stability without flexion strain.",
        "exercises": "McGill Modified Curl-Up; Bird Dog Holds; Side Bridge / Side Plank; Cat-Cow Mobilization; Gentle Hamstring Strap Stretch",
        "tags": "back pain; lumbar; spine; rehab; physical therapy; relief; beginner; core"
    },
    {
        "workout_id": 38,
        "title": "Indoor Rowing (Ergometer) 5k Cardio Challenge",
        "category": "Endurance",
        "difficulty_level": "Intermediate",
        "duration_minutes": 35,
        "target_muscles": "Full Body, Back, Quads, Cardiovascular",
        "equipment_needed": "Rowing Machine (Concept2 Erg)",
        "calories_burned_est": 430,
        "description": "Rhythm and power interval rowing session combining drive mechanics with steady split pacing for full-body cardiovascular and back endurance.",
        "exercises": "1000m Warmup Row; 5x500m Sprint Repeats (1:1 Rest); 2000m Steady Zone 3 Row; 500m Cool-down Paddle",
        "tags": "rowing; erg; cardio; full body; back; endurance; stamina; metabolic; indoor"
    },
    {
        "workout_id": 39,
        "title": "Advanced Calisthenics: Front Lever & Handstand Progression",
        "category": "Calisthenics",
        "difficulty_level": "Advanced",
        "duration_minutes": 50,
        "target_muscles": "Lats, Core, Shoulders, Wrists",
        "equipment_needed": "Pull-up Bar, Wall / Parallettes",
        "calories_burned_est": 310,
        "description": "Elite gymnastics strength training breaking down isometric skills: Tuck Front Lever holds, Handstand pushups, and Planche lean mechanics.",
        "exercises": "Chest-to-Wall Handstand Holds; Wall Handstand Pushups; Tuck Front Lever Holds; Dragon Flags; Scapular Pull-ups; Planche Leans",
        "tags": "calisthenics; handstand; front lever; gymnastics; advanced; bodyweight; core; shoulders"
    },
    {
        "workout_id": 40,
        "title": "TRX Suspension Full Body Sculpt",
        "category": "Strength",
        "difficulty_level": "Intermediate",
        "duration_minutes": 40,
        "target_muscles": "Core, Chest, Back, Hamstrings",
        "equipment_needed": "TRX Suspension Trainer",
        "calories_burned_est": 290,
        "description": "Suspension trainer workout utilizing gravity and body angles to enforce continuous core engagement across all movement patterns.",
        "exercises": "TRX Low Rows; TRX Chest Press; TRX Pistol Squat Assists; TRX Hamstring Curls; TRX Atomic Pushups; TRX Y-Flyes",
        "tags": "trx; suspension; core; functional; bodyweight; home gym; balance; stability"
    },
    {
        "workout_id": 41,
        "title": "Pre-Exhaust Hypertrophy Quad Blast",
        "category": "Hypertrophy",
        "difficulty_level": "Advanced",
        "duration_minutes": 55,
        "target_muscles": "Quadriceps, Vastus Medialis, Glutes",
        "equipment_needed": "Leg Extension Machine, Barbell, Leg Press",
        "calories_burned_est": 440,
        "description": "Hypertrophy technique isolating the quads with leg extensions before heavy compound squats, producing massive quad sweep with less spinal load.",
        "exercises": "Leg Extensions 4x15 (Pre-Exhaust); Hack Squat 4x10; Leg Press Foot Low 3x12; Sissy Squats 3x Failure; Walking Lunges",
        "tags": "quads; legs; hypertrophy; bodybuilding; pre-exhaust; leg press; machine; advanced"
    },
    {
        "workout_id": 42,
        "title": "HIIT Jump Rope Cardio Meltdown",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 25,
        "target_muscles": "Calves, Shoulders, Core, Heart",
        "equipment_needed": "Jump Rope",
        "calories_burned_est": 330,
        "description": "High-intensity jump rope intervals blended with bodyweight burpees and boxer skips to ignite rapid cardiovascular fat burn in 25 minutes.",
        "exercises": "Basic Bounce 60s; Boxer Skip 60s; Double Unders 30s; Burpees 30s; High Knees with Rope 45s; Cross-Overs 45s",
        "tags": "jump rope; skipping; hiit; cardio; fat burn; quick; calves; coordination; home"
    },
    {
        "workout_id": 43,
        "title": "Hamstring Hypertrophy & Knee Flexion Focus",
        "category": "Hypertrophy",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Biceps Femoris, Semitendinosus, Glutes",
        "equipment_needed": "Lying Leg Curl Machine, Dumbbells",
        "calories_burned_est": 320,
        "description": "Complete hamstring development incorporating both hip extension (RDLs) and knee flexion (leg curls) for full posterior chain aesthetics.",
        "exercises": "Lying Leg Curls 4x12; Romanian Dumbbell Deadlifts 4x10; Seated Leg Curls 3x15; Nordic Hamstring Curls 3x6; Glute Bridges",
        "tags": "hamstrings; legs; posterior chain; deadlift; curls; hypertrophy; knee health"
    },
    {
        "workout_id": 44,
        "title": "Kettlebell Hardstyle Snatch & Swing Test",
        "category": "Endurance",
        "difficulty_level": "Advanced",
        "duration_minutes": 35,
        "target_muscles": "Full Body, Posterior Chain, Grip, Heart",
        "equipment_needed": "Kettlebell",
        "calories_burned_est": 460,
        "description": "Pavel Tsatsouline inspired hardstyle kettlebell protocol for building unbreakable work capacity, hip snap speed, and mental resolve.",
        "exercises": "One-Arm Kettlebell Swings 10x10; Kettlebell Snatches 10x10; Clean and Strict Press 5x5; Halo Warmups; Goblet Squats",
        "tags": "kettlebell; hardstyle; snatch; swings; endurance; grip; conditioning; power"
    },
    {
        "workout_id": 45,
        "title": "Overhead Press Peak Strength & Lockout",
        "category": "Strength",
        "difficulty_level": "Advanced",
        "duration_minutes": 50,
        "target_muscles": "Anterior Deltoids, Upper Chest, Triceps, Core",
        "equipment_needed": "Barbell, Squat Rack",
        "calories_burned_est": 350,
        "description": "Strict military press focus aimed at building boulder shoulders, strong triceps, and iron standing core stability.",
        "exercises": "Strict Overhead Press 5x3; Push Press 3x3; Z-Press from Floor 3x8; Behind-the-Neck Press 3x10; Lateral Raises",
        "tags": "overhead press; military press; shoulders; strength; barbell; upper body; delts"
    },
    {
        "workout_id": 46,
        "title": "Bouldering & Climbing Specific Pull Conditioning",
        "category": "Calisthenics",
        "difficulty_level": "Intermediate",
        "duration_minutes": 45,
        "target_muscles": "Forearms, Lats, Finger Flexors, Core",
        "equipment_needed": "Hangboard, Pull-up Bar",
        "calories_burned_est": 320,
        "description": "Climbing-specific training session targeting finger tendon strength, lock-off endurance, and rotational core stability for climbers.",
        "exercises": "Hangboard Repeaters 7s/3s; Frenchies (Lock-off Pull-ups); Scapular Shrugs; L-Sit Holds; Offset Pull-ups; Rice Bucket Digs",
        "tags": "climbing; bouldering; grip; forearms; hangboard; pull-ups; lats; calisthenics"
    },
    {
        "workout_id": 47,
        "title": "Office Worker 10-Minute Lunchtime Energizer",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 10,
        "target_muscles": "Spine, Hips, Shoulders",
        "equipment_needed": "None / Chair",
        "calories_burned_est": 50,
        "description": "Quick 10-minute chair and standing routine designed for corporate professionals to boost circulation, open tight hips, and reset mental focus.",
        "exercises": "Seated Spinal Twists; Chair Squats; Standing Hip Circles; Doorway Chest Stretch; Standing Calf Raises; Deep Belly Breaths",
        "tags": "quick; 10 min; office; desk; lunchtime; chair; light; mobility; beginner"
    },
    {
        "workout_id": 48,
        "title": "Stationary Cycling / Spin Interval Blast",
        "category": "Cardio / HIIT",
        "difficulty_level": "Intermediate",
        "duration_minutes": 40,
        "target_muscles": "Quads, Hamstrings, Cardiovascular System",
        "equipment_needed": "Stationary Bike / Spin Bike",
        "calories_burned_est": 470,
        "description": "High-cadence cycling intervals alternating heavy resistance hill climbs with high-RPM sprints to torch calories without knee impact.",
        "exercises": "5-min Low Cadence Warmup; 8x30s High-Res Hill Climbs; 8x30s High-RPM Flat Sprints; 10-min Tempo Steady Cruise; 5-min Cool-down",
        "tags": "cycling; spin bike; bike; cardio; hiit; legs; low impact; stamina; fat burn"
    },
    {
        "workout_id": 49,
        "title": "Dumbbell German Volume Training (10x10 Mass)",
        "category": "Hypertrophy",
        "difficulty_level": "Advanced",
        "duration_minutes": 65,
        "target_muscles": "Chest, Back, Quads",
        "equipment_needed": "Dumbbells, Bench",
        "calories_burned_est": 500,
        "description": "Classic German Volume Training (GVT) system using 10 sets of 10 repetitions with fixed rest intervals for unprecedented hypertrophic stimulus.",
        "exercises": "Dumbbell Flat Bench Press 10x10; Dumbbell Romanian Deadlift 10x10; Dumbbell Goblet Squats 10x10; Dumbbell Hammer Curls 3x12",
        "tags": "gvt; german volume training; 10x10; hypertrophy; mass; dumbbells; intense; volume"
    },
    {
        "workout_id": 50,
        "title": "Full Body Functional Mobility & Joint Longevity",
        "category": "Yoga / Mobility",
        "difficulty_level": "Beginner",
        "duration_minutes": 30,
        "target_muscles": "Hips, Ankles, Thoracic Spine, Glenohumeral Joint",
        "equipment_needed": "Yoga Mat",
        "calories_burned_est": 120,
        "description": "Functional Range Conditioning (FRC) inspired joint mobility routine utilizing Controlled Articular Rotations (CARs) to expand usable range of motion.",
        "exercises": "Neck CARs; Shoulder CARs; Thoracic Spine CARs; Hip 90/90 Internal & External Rotations; Ankle CARs; Deep Squat Pry",
        "tags": "mobility; frc; joints; longevity; stretching; flexibility; hips; functional; recovery"
    }
]

# Write workout plans to CSV
df_workouts = pd.DataFrame(workouts)
workout_csv_path = os.path.join(DATA_DIR, "workout_plans.csv")
df_workouts.to_csv(workout_csv_path, index=False)
print(f"Saved {len(df_workouts)} workout plans to {workout_csv_path}")

# -------------------------------------------------------------
# 2. GENERATE FITNESS REVIEWS DATASET (120+ realistic reviews)
# -------------------------------------------------------------
reviews_data = [
    # Positive reviews
    {"review_id": 1, "workout_name": "Full Body HIIT Blast", "sentiment": "Positive", "rating": 5, "review_text": "This HIIT session completely kicked my butt! Burned over 350 calories and felt energised the whole day. Fantastic coach instructions."},
    {"review_id": 2, "workout_name": "Hypertrophy Push Day: Chest, Shoulders & Triceps", "sentiment": "Positive", "rating": 5, "review_text": "Incredible chest pump! The progressive overload scheme with barbell press and dips gave me noticeable upper body growth after four weeks."},
    {"review_id": 3, "workout_name": "PPL Pull Day: Back, Rear Delts & Biceps", "sentiment": "Positive", "rating": 5, "review_text": "Deadlifts and weighted pull-ups in the same session feel amazing. My lats and biceps are definitely wider and stronger."},
    {"review_id": 4, "workout_name": "Lower Body Quad & Glute Destroyer", "sentiment": "Positive", "rating": 5, "review_text": "Squats and lunges were brutal in the best way possible. Sore legs for two days but my squat PR went up by 10kg!"},
    {"review_id": 5, "workout_name": "Beginner Calisthenics & Bodyweight Mastery", "sentiment": "Positive", "rating": 5, "review_text": "As someone who could not do a single clean pushup, this progression guide was super clear, safe, and motivating."},
    {"review_id": 6, "workout_name": "Vinyasa Flow Mobility & Deep Flexibility", "sentiment": "Positive", "rating": 5, "review_text": "My lower back pain disappeared after doing this yoga flow twice a week. Hamstrings feel loose and hip mobility is transformed."},
    {"review_id": 7, "workout_name": "Starting Strength 5x5 Power Routine", "sentiment": "Positive", "rating": 5, "review_text": "The classic 5x5 compound lifts are unmatched for building true functional strength. Simple, effective, and results speak for themselves."},
    {"review_id": 8, "workout_name": "Core Sculpt & Six-Pack Shredder", "sentiment": "Positive", "rating": 5, "review_text": "Short 25-minute workout but absolute fire in the abs. Hanging leg raises and hollow body holds are killer."},
    {"review_id": 9, "workout_name": "Low-Impact Cardio for Seniors & Beginners", "sentiment": "Positive", "rating": 5, "review_text": "Gentle on my bad knees and very easy to follow at home. Great low impact cardio without joint ache."},
    {"review_id": 10, "workout_name": "Kettlebell Metabolic Circuit Shredder", "sentiment": "Positive", "rating": 5, "review_text": "The kettlebell swings and clean and press kept my heart rate in the fat-burning zone. Excellent conditioning!"},
    {"review_id": 11, "workout_name": "Upper Body Dumbbell Sculpt & Tone", "sentiment": "Positive", "rating": 5, "review_text": "Great dumbbell routine for home lifters. Arms and shoulders look sculpted and the pacing is very realistic."},
    {"review_id": 12, "workout_name": "Runner's Stamina & Aerobic Base Builder", "sentiment": "Positive", "rating": 5, "review_text": "Helped me shave 2 minutes off my 5k personal best. The track intervals build serious stamina and mental toughness."},
    {"review_id": 13, "workout_name": "Desk Worker Posture & Spinal Rehab", "sentiment": "Positive", "rating": 5, "review_text": "Immediate relief for my stiff neck and rounded shoulders. Wall angels and band pull-aparts should be mandatory for all programmers."},
    {"review_id": 14, "workout_name": "Chest Peak & Pectoral Mass Specialist", "sentiment": "Positive", "rating": 5, "review_text": "Outstanding pectoral hypertrophy workout. The cable crossovers and decline presses hit every angle of the chest."},
    {"review_id": 15, "workout_name": "Boxing Heavy Bag Conditioning & Agility", "sentiment": "Positive", "rating": 5, "review_text": "Best stress buster ever! Fast-paced rounds, intense calorie burn, and great combination cues from the trainer."},

    # Negative reviews
    {"review_id": 16, "workout_name": "Full Body HIIT Blast", "sentiment": "Negative", "rating": 1, "review_text": "Way too fast and chaotic. The transitions between burpees and mountain climbers caused severe wrist pain and dizzy spells."},
    {"review_id": 17, "workout_name": "Lower Body Quad & Glute Destroyer", "sentiment": "Negative", "rating": 1, "review_text": "Unreasonable volume for squats and split squats. My lower back was badly strained and the recovery took a whole week."},
    {"review_id": 18, "workout_name": "Hypertrophy Push Day: Chest, Shoulders & Triceps", "sentiment": "Negative", "rating": 2, "review_text": "Too many overlapping pressing exercises. Shoulder joints were clicking painfully on the incline press. Not balanced at all."},
    {"review_id": 19, "workout_name": "Starting Strength 5x5 Power Routine", "sentiment": "Negative", "rating": 2, "review_text": "Extremely repetitive and boring. Doing only squats and presses three times a week led to stalling and zero hypertrophy."},
    {"review_id": 20, "workout_name": "Core Sculpt & Six-Pack Shredder", "sentiment": "Negative", "rating": 1, "review_text": "Caused excessive neck strain during crunches and the lower back arches badly during leg raises. Poor form explanations."},
    {"review_id": 21, "workout_name": "Kettlebell Metabolic Circuit Shredder", "sentiment": "Negative", "rating": 2, "review_text": "The Turkish get-ups are far too complex for a fast paced circuit. Bruised my forearm with clumsy kettlebell transitions."},
    {"review_id": 22, "workout_name": "Tabata Sprint Extreme Fat Burner", "sentiment": "Negative", "rating": 1, "review_text": "Terrible for joints. Hard plyometric jumps on concrete floors gave me awful shin splints and knee inflammation."},
    {"review_id": 23, "workout_name": "Gymnastic Rings Upper Body Strength", "sentiment": "Negative", "rating": 2, "review_text": "Rings are too unstable and dangerous without a spotter. Elbow tendonitis flared up immediately during ring dips."},
    {"review_id": 24, "workout_name": "Runner's Stamina & Aerobic Base Builder", "sentiment": "Negative", "rating": 2, "review_text": "Pacing instructions are unrealistic for amateur runners. Ended up overtraining and pulling a calf muscle."},
    {"review_id": 25, "workout_name": "Upper Body Dumbbell Sculpt & Tone", "sentiment": "Negative", "rating": 2, "review_text": "Weights recommended were way too heavy for beginners. Zero modifications provided for people with limited dumbbell sets."},

    # Neutral reviews
    {"review_id": 26, "workout_name": "Full Body HIIT Blast", "sentiment": "Neutral", "rating": 3, "review_text": "Decent workout with average intensity. Good if you have only 30 minutes, but nothing special compared to other HIIT videos."},
    {"review_id": 27, "workout_name": "Vinyasa Flow Mobility & Deep Flexibility", "sentiment": "Neutral", "rating": 3, "review_text": "Standard yoga flow. Some poses were relaxing, but the audio quality was a bit muffled and background music was distracting."},
    {"review_id": 28, "workout_name": "PPL Pull Day: Back, Rear Delts & Biceps", "sentiment": "Neutral", "rating": 3, "review_text": "Standard pull day split. Exercises are basic staple movements; works fine but lacks novelty or unique variations."},
    {"review_id": 29, "workout_name": "Low-Impact Cardio for Seniors & Beginners", "sentiment": "Neutral", "rating": 3, "review_text": "Okay for total beginners or seniors. Might be too mild for people who already walk regularly or have some baseline fitness."},
    {"review_id": 30, "workout_name": "Desk Worker Posture & Spinal Rehab", "sentiment": "Neutral", "rating": 3, "review_text": "Exercises are okay for posture maintenance, but you need resistance bands and foam roller handy. Average routine overall."},
    {"review_id": 31, "workout_name": "Arm Farm: Biceps & Triceps Super-Sets", "sentiment": "Neutral", "rating": 3, "review_text": "Gives a nice temporary arm pump, but supersets take up two stations at a crowded commercial gym."},
    {"review_id": 32, "workout_name": "Beginner Calisthenics & Bodyweight Mastery", "sentiment": "Neutral", "rating": 3, "review_text": "Good fundamental movements, but progression between knee push-ups and standard pushups is quite steep for true beginners."},
    {"review_id": 33, "workout_name": "Rest Day Active Recovery & Foam Rolling", "sentiment": "Neutral", "rating": 3, "review_text": "Routine is quite basic. You could just do regular stretching, but the foam rolling guidance is mildly helpful."},
    {"review_id": 34, "workout_name": "Kettlebell Metabolic Circuit Shredder", "sentiment": "Neutral", "rating": 3, "review_text": "Requires a solid kettlebell technique before starting. It is an okay cardio substitute if the weather prevents outdoor runs."},
    {"review_id": 35, "workout_name": "Total Body Resistance Band Home Circuit", "sentiment": "Neutral", "rating": 3, "review_text": "Fine for business travel in a hotel room, but does not replace heavy barbell lifting if strength is your main goal."}
]

# Expand dataset with more variations to give 100+ high quality rows for training and evaluation
expanded_reviews = []
sentiment_templates = {
    "Positive": [
        "Brilliant workout plan for {goal}. I noticed increased muscular endurance and stamina within 3 weeks.",
        "Loved the clear trainer instructions for {exercise}. Highly recommended for anyone wanting results.",
        "Super effective workout! My heart rate stayed high and calorie burn was phenomenal.",
        "Excellent programming. Exercises target the right muscle groups without unnecessary joint strain.",
        "One of the best routines I have tried. Great structure, clear rest intervals, and fantastic pump.",
        "Very engaging and intense. Really helped me stay disciplined on my fat loss journey.",
        "The pacing was spot-on and the exercise form cues helped prevent any back discomfort."
    ],
    "Negative": [
        "Disappointed with the routine. The exercise tempo is way too fast and form breaks down easily.",
        "Felt painful in my shoulders and knees. Not beginner friendly at all despite the description.",
        "Poor instructions and confusing exercise sequence. Ended up with terrible muscular soreness and fatigue.",
        "Waste of time. Did not feel any real engagement in my target muscles and pacing was frustrating.",
        "Far too difficult for the stated level. Almost injured my lower back on the very first set.",
        "Lack of proper warm-up guidelines. My joints were screaming after just fifteen minutes.",
        "Way too many complex movements packed into a short time window. Very sloppy execution."
    ],
    "Neutral": [
        "An average workout session. It is okay if you are short on time, but nothing extraordinary.",
        "Standard exercises that get the job done. Neither particularly exciting nor particularly bad.",
        "Decent routine, but you definitely need good gym equipment to perform most of the movements.",
        "The exercises are fine, but the audio pacing and rest times feel a bit inconsistent.",
        "Moderate intensity workout. Good as a supplement, but would not use it as my primary routine.",
        "Fairly typical fitness circuit. Good for maintenance, but progress was rather slow."
    ]
}

goals = ["fat loss", "muscle growth", "mobility", "strength gains", "conditioning", "stamina"]
sample_exercises = ["squats", "bench press", "pushups", "deadlifts", "kettlebell swings", "hiit sprints"]

id_counter = 36
for sent, templates in sentiment_templates.items():
    for template in templates:
        for w in workouts[:6]:
            text = template.replace("{goal}", goals[id_counter % len(goals)]).replace("{exercise}", sample_exercises[id_counter % len(sample_exercises)])
            rating = 5 if sent == "Positive" else (1 if sent == "Negative" else 3)
            expanded_reviews.append({
                "review_id": id_counter,
                "workout_name": w["title"],
                "sentiment": sent,
                "rating": rating,
                "review_text": text
            })
            id_counter += 1

all_reviews = reviews_data + expanded_reviews
df_reviews = pd.DataFrame(all_reviews)
reviews_csv_path = os.path.join(DATA_DIR, "fitness_reviews.csv")
df_reviews.to_csv(reviews_csv_path, index=False)
print(f"Saved {len(df_reviews)} fitness reviews to {reviews_csv_path}")

# -------------------------------------------------------------
# 3. GENERATE FITNESS ENTITIES DICTIONARY
# -------------------------------------------------------------
entities = {
    "EXERCISE": [
        "squat", "squats", "bench press", "deadlift", "deadlifts", "overhead press",
        "barbell row", "pushup", "pushups", "push-up", "push-ups", "pullup", "pullups",
        "pull-up", "pull-ups", "bicep curl", "bicep curls", "hammer curl", "tricep dip",
        "dips", "lunges", "walking lunges", "bulgarian split squat", "hip thrust",
        "kettlebell swing", "kettlebell swings", "burpee", "burpees", "mountain climber",
        "mountain climbers", "jumping jacks", "plank", "side plank", "russian twist",
        "lat pulldown", "cable fly", "face pull", "leg press", "leg extension",
        "hamstring curl", "calf raise", "clean and jerk", "snatch", "chin up", "running",
        "sprinting", "jogging", "cycling", "rowing", "stretching"
    ],
    "MUSCLE_GROUP": [
        "chest", "pectoral", "pecs", "back", "lats", "latissimus dorsi", "traps", "trapezius",
        "shoulders", "delts", "deltoids", "biceps", "triceps", "forearms", "abs", "abdominals",
        "core", "obliques", "quads", "quadriceps", "hamstrings", "glutes", "butt", "calves",
        "legs", "upper body", "lower body", "posterior chain", "full body", "spine"
    ],
    "WORKOUT_TYPE": [
        "hiit", "high intensity interval training", "strength training", "hypertrophy",
        "muscle building", "bodybuilding", "powerlifting", "calisthenics", "bodyweight",
        "yoga", "vinyasa", "pilates", "cardio", "aerobic", "tabata", "circuit training",
        "crossfit", "metabolic conditioning", "stretching", "mobility", "rehabilitation"
    ],
    "DIFFICULTY_LEVEL": [
        "beginner", "novice", "intermediate", "medium", "advanced", "expert", "pro"
    ],
    "EQUIPMENT": [
        "dumbbell", "dumbbells", "barbell", "kettlebell", "kettlebells", "pull-up bar",
        "pullup bar", "bench", "weight plates", "resistance band", "resistance bands",
        "rings", "gymnastic rings", "cable machine", "cables", "foam roller", "mat",
        "yoga mat", "treadmill", "heavy bag", "bodyweight", "no equipment"
    ],
    "GOAL": [
        "weight loss", "fat loss", "burn fat", "lose weight", "muscle gain", "build muscle",
        "hypertrophy", "strength", "get stronger", "stamina", "endurance", "flexibility",
        "mobility", "six pack", "posture", "toned body", "recovery"
    ]
}

entities_json_path = os.path.join(DATA_DIR, "fitness_entities.json")
with open(entities_json_path, "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=4)
print(f"Saved fitness entities to {entities_json_path}")

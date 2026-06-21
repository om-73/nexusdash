const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5001/api';
let token = '';

async function runTests() {
  console.log('=== STARTING NEXUSDASH E2E API VERIFICATION ===');
  
  try {
    // 1. Authenticate
    console.log('\n[1/10] Authenticating default admin user...');
    const loginRes = await axios.post(`${BASE_URL}/auth/login`, {
      email: 'admin@nexusdash.com',
      password: 'admin123'
    });
    token = loginRes.data.token;
    console.log('✔ Authenticated successfully. Token retrieved.');

    const headers = { Authorization: `Bearer ${token}` };

    // 2. Upload Dummy Data File
    console.log('\n[2/10] Simulating file upload...');
    const dummyPath = path.resolve(__dirname, '../dummy_data.csv');
    const uploadDirPath = path.resolve(__dirname, './uploads');
    const targetPath = path.join(uploadDirPath, 'test_dummy_data.csv');
    
    // Copy dummy_data.csv to uploads folder
    if (!fs.existsSync(uploadDirPath)) {
      fs.mkdirSync(uploadDirPath);
    }
    fs.copyFileSync(dummyPath, targetPath);
    console.log(`✔ Copied dummy_data.csv to uploads folder at: ${targetPath}`);

    // 3. Load Dataset into Data Engine
    console.log('\n[3/10] Loading dataset via /api/data/load...');
    const loadRes = await axios.post(`${BASE_URL}/data/load`, {
      file_path: targetPath,
      file_type: 'csv'
    }, { headers });
    console.log('✔ Dataset loaded successfully. Metadata keys:', Object.keys(loadRes.data));
    console.log('Columns found:', loadRes.data.columns);
    console.log('Total rows:', loadRes.data.rows);

    // 4. Verify Dataset State
    console.log('\n[4/10] Checking active dataset state via /api/data/state...');
    const stateRes = await axios.get(`${BASE_URL}/data/state`, { headers });
    console.log('✔ State verification: active dataset columns are:', stateRes.data.columns);

    // 5. Clean Dataset
    console.log('\n[5/10] Performing cleaning operation /api/data/clean (dropna)...');
    const cleanRes = await axios.post(`${BASE_URL}/data/clean`, {
      operation: 'dropna',
      columns: ['feature1', 'feature2']
    }, { headers });
    console.log('✔ Cleaning completed. Rows after dropna:', cleanRes.data.rows);

    // 6. Perform Exploratory Data Analysis (EDA)
    console.log('\n[6/10] Fetching EDA results via /api/data/eda...');
    const edaRes = await axios.get(`${BASE_URL}/data/eda`, { headers });
    console.log('✔ EDA completed. Feature list:', Object.keys(edaRes.data.summary || edaRes.data));

    // 7. Calculate KPI
    console.log('\n[7/10] Calculating KPI via /api/data/kpi/calculate...');
    const kpiRes = await axios.post(`${BASE_URL}/data/kpi/calculate`, {
      column: 'target',
      operation: 'mean'
    }, { headers });
    console.log('✔ KPI calculate completed:', kpiRes.data);

    // 8. Apply Feature Engineering
    console.log('\n[8/10] Performing feature engineering (quantiles/conditional) via /api/data/feature/engineer...');
    const engineerRes = await axios.post(`${BASE_URL}/data/feature/engineer`, {
      features: [
        {
          name: 'feature_interaction',
          type: 'interaction',
          config: {
            col1: 'feature1',
            col2: 'feature2',
            operator: 'multiply'
          }
        }
      ]
    }, { headers });
    console.log('✔ Feature engineering completed. New columns:', engineerRes.data.columns);

    // 9. Train Model
    console.log('\n[9/10] Training Machine Learning model via /api/data/train...');
    const trainRes = await axios.post(`${BASE_URL}/data/train`, {
      feature_columns: ['feature1', 'feature2', 'feature_interaction'],
      target_column: 'target',
      problem_type: 'regression',
      algorithms: ['linear', 'rf'],
      test_size: 0.2
    }, { headers });
    console.log('✔ Model trained successfully!');
    console.log('Best algorithm:', trainRes.data.best_algorithm);
    console.log('Metrics:', trainRes.data.metrics);
    console.log('Actual preview vs Predicted preview:');
    console.log('Actuals:   ', trainRes.data.actual_preview);
    console.log('Predictions:', trainRes.data.predictions_preview);

    // 10. Predict Model
    console.log('\n[10/10] Testing prediction endpoint /api/data/predict...');
    const predictRes = await axios.post(`${BASE_URL}/data/predict`, {
      inputs: {
        feature1: 3.5,
        feature2: 4.5,
        feature_interaction: 15.75
      }
    }, { headers });
    console.log('✔ Prediction result:', predictRes.data);

    // 11. AI Assistant Chat Check
    console.log('\n[BONUS] Testing AI Assistant Chat widget /api/data/ai/chat...');
    const chatRes = await axios.post(`${BASE_URL}/data/ai/chat`, {
      messages: [
        { role: 'user', content: 'What is the target column in the loaded dataset?' }
      ],
      systemInstruction: 'You are Nexus AI, a helpful data assistant. The target column is target.'
    }, { headers });
    console.log('✔ Chat Widget Response:', chatRes.data.reply);

    console.log('\n=== ALL E2E BACKEND TESTS COMPLETED SUCCESSFULLY! ===\n');
  } catch (error) {
    console.error('\n❌ E2E Verification failed with error:');
    if (error.response) {
      console.error(`Status: ${error.response.status}`);
      console.error('Data:', JSON.stringify(error.response.data, null, 2));
    } else {
      console.error(error.message);
    }
    process.exit(1);
  }
}

runTests();

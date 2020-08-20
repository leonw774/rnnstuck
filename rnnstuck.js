const vocabSize = WORD_INDEX.length;
const seedSize = SEED_INDEX.length;
const ZERO_VECTOR = new Array(VECTOR_INDEX[0].length).fill(0.0)
const OUTPUT_MAX_TIMESTEP = 64;
const SAMPLE_TEMPERATURE = 0.5;
const END_MARK = "ê"

var model_loaded = false;
var max_model_timestep = null;
model = null;

var model_status_div = document.getElementById("model-status-div"),
    gen_btn = document.getElementById("gen-btn"),
    gen_div = document.getElementById("gen-div"),
    gen_st = document.getElementById("gen-status");

async function load_model() {
  model_status_div.innerText = "........正在載入模型........";
  //model = await tf.loadLayersModel('./jsmodel/model.json');
  model = await tf.loadLayersModel('https://leonw774.github.io/rnnstuck/jsmodel/model.json');
  max_model_timestep = model.layers[0].inputSpec[0].shape[1];
  model_loaded = true;
  model_status_div.innerText = "模型載入完成。";
  gen_btn.disabled = false;
  return;
};

function multinomial(probs) {
  let pmax = 0.0;
  let acc_prob = [];
  for(let i in probs) {
    pmax += probs[i];
    acc_prob.push(pmax);
  }
  let r = Math.random() * pmax;
  for (let i in acc_prob) {
    if (r <= acc_prob[i])
      return i;
  }
}; 

function sample(prediction, temperature = 1.0) {
  // prediction is a array of probability
  let sum = 0.0;
  for (let i in prediction) {
    prediction[i] = Math.exp((Math.log(prediction[i]) / temperature));
    sum += prediction[i];
  }
  for (let i in prediction)
    prediction[i] /= sum;
  return multinomial(prediction);
};

function word2vec(word) {
  let i = WORD_INDEX.findIndex(x => x == word);
  // let i = VECTOR[word];
  if (i != -1) {
    return VECTOR_INDEX[i];
  }
  else {
    return ZERO_VECTOR;
  }
};

async function generate() {
  if (!model_loaded) return;
  
  gen_div.innerText = "";
  gen_st.innerText = "........正在產生文字........";
  gen_btn.disabled = true;
  
  let output_sentence = [SEED_INDEX[Math.floor(Math.random() * seedSize)]];
  let next_word = output_sentence[0];
  let sentence_array = new Array();
  sentence_array.push(new Array(max_model_timestep));
  sentence_array[0].fill(ZERO_VECTOR);
  
  let time_static = [];
  let latest_dt = Date.now();
  
  for (let i = 0; i < OUTPUT_MAX_TIMESTEP; i++) {
    if (i < max_model_timestep) {
      sentence_array[0][i] = word2vec(next_word)
    }
    else {
      sentence_array[0].shift();
      sentence_array[0].push(word2vec(next_word));
    }
    
    let y_data = Array.from(model.predict(tf.tensor(sentence_array)).dataSync());
    next_word = WORD_INDEX[sample(y_data, SAMPLE_TEMPERATURE)];
    
    if (next_word == END_MARK || next_word == undefined) break;
    output_sentence.push(next_word);
    
    time_static.push(Date.now() - latest_dt);
    latest_dt = Date.now();
  }
  
  let sum = 0;
  for (let i in time_static) {
    sum += time_static[i];
  }
  console.log("average loop time:", sum/time_static.length);
  
  gen_btn.disabled = false;
  gen_div.innerText = output_sentence.join("");
  gen_st.innerText = "";
}

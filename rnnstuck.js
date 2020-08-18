const vocabSize = WORD_INDEX.length;
const seedSize = SEED_INDEX.length;
const ZERO_VECTOR = new Array(VECTOR_DICT["\n"].length).fill(0.0)
const OUTPUT_MAX_TIMESTEP = 100;
const SAMPLE_TEMPERATURE = 0.75;
const END_MARK = "ê"

var model_loaded = false;
var max_timestep = null;
model = null;

var model_status_div = document.getElementById("model-status-div"),
    gen_btn = document.getElementById("gen-btn"),
    gen_div = document.getElementById("gen-div"),
    gen_st = document.getElementById("gen-status");

async function load_model() {
  model_status_div.innerText = "........正在載入模型........";
  //model = await tf.loadLayersModel('./jsmodel/model.json');
  model = await tf.loadLayersModel('https://leonw774.github.io/rnnstuck/jsmodel/model.json');
  max_timestep = model.layers[0].inputSpec[0].shape[1];
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
  let found = VECTOR_DICT[word];
  if (found != null) {
    return found;
  }
  else {
    return ZERO_VECTOR;
  }
};

function sentence2vecs(sentence) {
  let result = [];
  let sentence_in = sentence;
  if (max_timestep != null) {
    if (sentence.length > max_timestep) {
      sentence_in = sentence.slice(sentence.length-max_timestep);
    }
  }
  for (let n = 0; n < max_timestep; n++) {
    if (n >= sentence_in.length) {
      result.push(ZERO_VECTOR);
    }
    else {
      result.push(word2vec(sentence_in[n]))
    }
  }
  return [result];
};

async function generate() {
  if (!model_loaded) return;
  
  gen_div.innerText = "";
  gen_st.innerText = "........正在產生文字........";
  gen_btn.disabled = true;
  
  let output_sentence = [SEED_INDEX[Math.floor(Math.random() * seedSize)]];
  let sentence_vec = sentence2vecs(output_sentence);
  let next_word = "";
  
  let time_static = [];
  let latest_dt = Date.now();
  
  for (let i = 0; i < OUTPUT_MAX_TIMESTEP; i++) {
    tf.tidy(() => {
        let y_data = Array.from(model.predict(tf.tensor(sentence_vec)).dataSync());
        next_word = WORD_INDEX[sample(y_data, SAMPLE_TEMPERATURE)];
    });
    if (next_word == END_MARK) break;
    output_sentence.push(next_word);
    sentence_vec.push(word2vec(next_word));
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

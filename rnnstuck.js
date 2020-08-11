const vocabSize = WORD_INDEX.length;
const seedSize = SEED_INDEX.length;
const ZERO_VECTOR = new Array(VECTOR_INDEX[0].length).fill(0.0)
const OUTPUT_MAX_TIMESTEP = 100;
const SAMPLE_TEMPERATURE = 0.8;
const END_MARK = "ê"
var model_loaded = false;
var max_timestep = null;
model = null;


async function load_model() {
  var gen_btn = document.getElementById("gen-btn"),
      load_btn = document.getElementById("load-btn"),
      out_div = document.getElementById("output-div");
  load_btn.disabled = true;
  out_div.innerText = "........正在載入模型........";
  //model = await tf.loadLayersModel('./jsmodel/model.json');
  model = await tf.loadLayersModel('https://leonw774.github.io/rnnstuck/jsmodel/model.json');
  max_timestep = model.layers[0].inputSpec[0].shape[1];
  model_loaded = true;
  out_div.innerText = "模型載入完成。";
  gen_btn.disabled = false;
  load_btn.style.display = "none";
  return;
};

function multinomial(probs) {
  var l = probs.length, pmax = 0.0;
  var acc_prob = [];
  for(var i in probs) {
    pmax += probs[i];
    acc_prob.push(pmax);
  }
  var r = Math.random() * pmax;
  for (var i in acc_prob) {
    if (r <= acc_prob[i])
      return i;
  }
}; 

function sample(prediction, temperature = 1.0) {
  // prediction is a array of probability
  var sum = 0.0;
  for (var i in prediction) {
    prediction[i] = Math.exp((Math.log(prediction[i]) / temperature));
    sum += prediction[i];
  }
  for (var i in prediction)
    prediction[i] /= sum;
  return multinomial(prediction);
};

function sentence2vecs(sentence) {
  var result = [];
  var sentence_in = sentence;
  if (max_timestep != null) {
    if (sentence.length > max_timestep) {
      sentence_in = sentence.slice(sentence.length-max_timestep);
    }
  }
  for (var n = 0; n < max_timestep; n++) {
    if (n >= sentence_in.length) {
      result.push(ZERO_VECTOR);
    }
    else {
      var w = sentence_in[n];
      var i = WORD_INDEX.findIndex(element => element == sentence_in[w])
      if (i != -1) {
        result.push(VECTOR_INDEX[i]);
      }
      else {
        result.push(ZERO_VECTOR);
      }
    }
  }
  return [result];
};

async function generate() {
  if (!model_loaded) return;
  
  var gen_btn = document.getElementById("gen-btn"),
       gen_div = document.getElementById("gen-div"),
       gen_st = document.getElementById("gen-status");
  gen_div.innerText = "";
  gen_st.innerText = "........正在產生文字........";
  gen_btn.disabled = true;
  
  var output_sentence = [SEED_INDEX[Math.floor(Math.random() * seedSize)]];
  var next_word = "", last_word = "";
  /*
  var time_static = [];
  var latest_dt = Date.now();
  */
  for (var i = 0; i < OUTPUT_MAX_TIMESTEP; i++) {
    tf.tidy(() => {
        var y_data = Array.from(model.predict(tf.tensor(sentence2vecs(output_sentence))).dataSync());
        next_word = WORD_INDEX[sample(y_data, SAMPLE_TEMPERATURE)];
    });
    if (next_word == END_MARK) break;
    //if (last_word == "\n" && next_word == "\n") continue;
    output_sentence.push(next_word);
    last_word = next_word;
    //time_static.push(Date.now() - latest_dt);
    //latest_dt = Date.now();
  }
  /*
  var sum = 0;
  for (var i in time_static) {
    sum += time_static[i];
  }
  console.log("average loop time:", sum/time_static.length);
  */
  gen_btn.disabled = false;
  gen_div.innerText = output_sentence.join("");
  gen_st.innerText = "";
}

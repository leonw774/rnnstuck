const vocabSize = WORD_INDEX.length;
const seedSize = SEED_INDEX.length;
const ZERO_VECTOR = new Array(VECTOR_INDEX[0].length).fill(0.0)
var model_loaded = false;
var max_timestep = null;
model = null;


async function load_model() {
  var gen_btn = document.getElementById("gen-btn"),
      load_btn = document.getElementById("load-btn"),
      out_div = document.getElementById("output-div");
  load_btn.disabled = true;
  out_div.innerText = "........正在載入model........";
  //model = await tf.loadLayersModel('./jsmodel/model.json');
  model = await tf.loadLayersModel('https://leonw774.github.io/rnnstuck/jsmodel/model.json');
  max_timestep = model.layers[0].inputSpec[0].shape[1];
  model_loaded = true;
  out_div.innerText = "model載入完成。";
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
  if (max_timestep != null) {
    if (sentence.length > max_timestep) {
      sentence = sentence.slice(sentence.length-max_timestep);
    }
  }
  for (var n = 0; n < max_timestep; n++) {
    if (n >= sentence.length) {
      result.push(ZERO_VECTOR);
    }
    else {
      var w = sentence[n];
      var i = WORD_INDEX.findIndex(element => element == sentence[w])
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
  for (var i = 0; i < 60; i++) {
    tf.tidy(() => {
        var vector = sentence2vecs(output_sentence);
        console.log(vector.length);
        var y_data = Array.from(model.predict(tf.tensor(vector)).dataSync());
        next_word = WORD_INDEX[sample(y_data, 0.75)];
        y_data = [];
    });
    if (next_word == "ê") break;
    //if (last_word == "\n" && next_word == "\n") continue;
    output_sentence.push(next_word);
    last_word = next_word;
  }
  var output_string = "";
  for (var i in output_sentence) {
    output_string = output_string.concat(output_sentence[i]);
  }
  gen_btn.disabled = false;
  gen_div.innerText = output_string;
  gen_st.innerText = "";
}

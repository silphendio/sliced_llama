function $(query){ return document.querySelector(query) }
function el(name, attributes = {}, children = []){
  let el = document.createElement(name)
  for(let a in attributes){
    el.setAttribute(a, attributes[a])
  }
  el.append(...children)
  return el
}

update_model_info()



// may throw exception for malformed input
function get_layer_list(slices_str){
  layer_list = []
  for(slice of slices_str.trim().split(',')){
    r = slice.trim().split('-')
    for(let i = Number(r[0]); i < Number(r[1]); ++i){
      layer_list.push(i)
    }
  }
  console.log('layers: ', layer_list)
  return layer_list
}

function update_layers(){
  const response = fetch("/v1/rearrange_layers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      "layer_list": get_layer_list($("#layers").value)
    })
  })
  response.then(res => console.log(res))
  // TODO: incidate success on webpage

}
async function load_model(){
  const result = await fetch("/v1/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: $("#model_file").value,
      cache_mode: $("#cache_mode").value,
      max_seq_len: $("#max_seq_len").value == "" ? null : $("#max_seq_len").valueAsNumber
    })
  }).then(res => res.json())
  if(result.success){
    update_model_info()
  }
  else{
    console.log(result)
    $('#model_info').innerHTML = "<b>Error: " + result.detail + "</b>"
  }
}

async function update_model_info(){
  const model_info = await fetch('/v1/internal/model/info').then(res => res.json())
    $('#model_info').innerHTML = model_info.id ? "Model:" + model_info.id : "No model loaded..."
    //$('#model_file').value = model_info.id
    $('#num_layer_span').innerText = model_info.available_layers
    // TODO: fetch & update generation parameters, used layers, ...

}

var event_source
async function stream_text(){
  try{ stop_strings = JSON.parse('[' + $("#stop_strings").value + ']')}
  catch(e){stop_strings = []}

  event_source = new SSE("/v1/completions", {
    headers: {'Content-Type': 'application/json'},
    payload: JSON.stringify({
      //prompt: e('completions_area').value,
      prompt: $('#completions_area').innerText,
      logprobs: $('#logprobs').valueAsNumber,
      max_tokens: $('#max_tokens').valueAsNumber,
      stream: true,
      token_repetition_penalty: $('#token_repetition_penalty').valueAsNumber,
      token_repetition_range: $('#token_repetition_range').valueAsNumber,
      token_repetition_decay: $('#token_repetition_decay').valueAsNumber,
      temperature: $('#temperature').valueAsNumber,
      temperature_last:  $('#temperature_last').checked,
      top_k: $('#top_k').valueAsNumber,
      top_p: $('#top_p').valueAsNumber,
      min_p: $('#min_p').valueAsNumber,
      tfs: $('#tfs').valueAsNumber,
      typical: $('#typical').valueAsNumber,
      mirostat: $('#mirostat').checked,
      mirostat_tau: $('#mirostat_tau').valueAsNumber,
      mirostat_eta: $('#mirostat_eta').valueAsNumber,
      stop: stop_strings,
    }),
    method: 'POST',
  })
  console.log(event_source.payload)

  // create logprobs element
  var el_logrobs = el("div", )

  event_source.addEventListener('message', event => {
    res = JSON.parse(event.data)

    // calculate logprobs
    let logprobs = res.choices[0].logprobs.top_logprobs[0]
    if(res.choices[0].text.includes("digit")) console.log(event.data)
    let prob_labels = []
    // TODO: sort logprobs
    for(token in logprobs){
      let prob_percent = (Math.exp(logprobs[token])*100).toLocaleString(undefined, { minimumFractionDigits: 2 })
      prob_labels.push(
        el("div",
          {class:"token_prob"},
          [JSON.stringify(token).slice(1,-1), el("br"), prob_percent + "%"]
        )
      )
    }
    let logprobs_el = el("div", {}, prob_labels)

    // token element
    let token_text = res.choices[0].text.replace('\n', ' ')
    let token_element = el("span", { class: "token" }, token_text)
    
    $('#completions_area').append(token_element)

    // html is weird about newlines, even with white-space: pre-wrap
    if(res.choices[0].text.includes('\n')){
      $('#completions_area').appendChild(el('br'))
    }

    // tooltip
    token_element.addEventListener("mouseover", () => display_tooltip(token_element, logprobs_el))
    token_element.addEventListener("mouseout", hide_tooltip)
  })
  event_source.stream()
}

function stop_generation(){
  if(event_source) event_source.close()
}

function display_tooltip(parent, element){
  let rect = parent.getBoundingClientRect()
  let vp = visualViewport
  
  let tooltip_el = $("#tooltip")
  tooltip_el.innerHTML = ""
  console.log("visualViewport", visualViewport)
  console.log("rect", rect)
  
  if(rect.bottom + 80 < visualViewport.height){
    tooltip_el.style.top = (rect.bottom + 10) + "px"
    tooltip_el.style.bottom = ""
  }
  else {
    tooltip_el.style.bottom = (visualViewport.height - rect.top + 10) + "px"
    tooltip_el.style.top = ""
  }
  // TODO: better calculate x (make sure it's always entirely inside)
  //tooltip_el.style.left = (rect.x) + "px"
  tooltip_el.append(element)
  $("#tooltip").style.display="block"
}
function hide_tooltip() {
  $("#tooltip").style.display="none"
}

function toggle_monospace(element) {
  console.log(element)
  if(element.checked){
    $("#completions_area").style.fontFamily = "monospace, monospace"
  }
  else {
    $("#completions_area").style.fontFamily = ""
  }
}

function toggle_wrap_lines(element){
  console.log(element.checked)
  if(element.checked){
    $("#completions_area").style.whiteSpace = "pre-wrap"
  }
  else {
    $("#completions_area").style.whiteSpace = "pre"
  }
  console.log($("#completions_area").style.whiteSpace)
}

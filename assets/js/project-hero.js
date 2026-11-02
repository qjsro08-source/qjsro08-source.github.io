// Step-through hero for a hand-built project page.
//
// Markup: an element with [data-hero] holding step buttons [data-step-btn],
// each with how long its step holds (data-hold, ms). Optional pieces:
//   readout      [data-out-name] [data-out-acc] [data-out-sub], fed by the
//                button's data-name, data-acc, data-dec (decimals), data-sub
//   [data-at=i]  gets is-past / is-now / is-next relative to the step
//   [data-step-text="a|b|c"]   text for each step
//   [data-step-width="1,2,3"]  width (%) for each step
// CSS keys everything else off the hero's data-step attribute, and the
// markup ships the final step, so the page reads correctly without this file.
(function () {
  var hero = document.querySelector('[data-hero]')
  if (!hero) return

  var btns = Array.prototype.slice.call(hero.querySelectorAll('[data-step-btn]'))
  var toggle = hero.querySelector('[data-toggle]')
  var replay = hero.querySelector('[data-replay]')
  var outName = hero.querySelector('[data-out-name]')
  var outAcc = hero.querySelector('[data-out-acc]')
  var outSub = hero.querySelector('[data-out-sub]')
  var last = btns.length - 1
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches

  var step = last
  var timer = null
  var playing = false    // the user's choice: autoplay on or off
  var visible = true     // on screen and in a foreground tab
  var shownAcc = outAcc ? parseFloat(btns[last].dataset.acc) : 0
  var atEls = Array.prototype.slice.call(hero.querySelectorAll('[data-at]'))
  var textEls = Array.prototype.slice.call(hero.querySelectorAll('[data-step-text]'))
  var widthEls = Array.prototype.slice.call(hero.querySelectorAll('[data-step-width]'))
  var countFrame = null
  var countSettle = null

  function countTo(target, dec) {
    if (!outAcc) return
    if (countFrame) cancelAnimationFrame(countFrame)
    clearTimeout(countSettle)
    if (reduce) { outAcc.textContent = target.toFixed(dec); shownAcc = target; return }
    var from = shownAcc, t0 = null, dur = 650
    // Frames can be throttled (background tab), so land on the exact value regardless.
    countSettle = setTimeout(function () {
      if (countFrame) cancelAnimationFrame(countFrame)
      shownAcc = target
      outAcc.textContent = target.toFixed(dec)
    }, dur + 60)
    function tick(t) {
      if (t0 === null) t0 = t
      var k = Math.min(1, (t - t0) / dur)
      var e = 1 - Math.pow(1 - k, 3)
      shownAcc = from + (target - from) * e
      outAcc.textContent = shownAcc.toFixed(dec)
      if (k < 1) countFrame = requestAnimationFrame(tick)
    }
    countFrame = requestAnimationFrame(tick)
  }

  function show(i) {
    step = i
    var b = btns[i]
    hero.dataset.step = String(i)
    hero.style.setProperty('--hold', (b.dataset.hold || 2000) + 'ms')
    btns.forEach(function (x, j) { x.setAttribute('aria-pressed', j === i ? 'true' : 'false') })
    if (outName) outName.textContent = b.dataset.name
    if (outSub) outSub.textContent = b.dataset.sub
    countTo(parseFloat(b.dataset.acc), parseInt(b.dataset.dec || '1', 10))
    atEls.forEach(function (el) {
      var k = parseInt(el.dataset.at, 10)
      el.classList.toggle('is-past', k < i)
      el.classList.toggle('is-now', k === i)
      el.classList.toggle('is-next', k > i)
    })
    textEls.forEach(function (el) { el.textContent = el.dataset.stepText.split('|')[i] })
    widthEls.forEach(function (el) { el.style.width = el.dataset.stepWidth.split(',')[i] + '%' })
    // Restart the progress bar on the active step.
    hero.classList.remove('is-ticking')
    void hero.offsetWidth
    if (playing && visible) hero.classList.add('is-ticking')
  }

  function schedule() {
    clearTimeout(timer)
    if (!playing || !visible) return
    timer = setTimeout(function () {
      show(step === last ? 0 : step + 1)
      schedule()
    }, parseInt(btns[step].dataset.hold || '2000', 10))
  }

  function setPlaying(on) {
    playing = on
    hero.classList.toggle('is-paused', !on)
    toggle.textContent = on ? '일시정지' : '재생'
    toggle.setAttribute('aria-label', on ? '애니메이션 일시정지' : '애니메이션 재생')
    if (on && visible) hero.classList.add('is-ticking')
    else hero.classList.remove('is-ticking')
    schedule()
  }

  btns.forEach(function (b, i) {
    b.addEventListener('click', function () { setPlaying(false); show(i) })
  })
  toggle.addEventListener('click', function () {
    if (!playing && step === last) show(0)
    setPlaying(!playing)
  })
  replay.addEventListener('click', function () { show(0); setPlaying(true) })

  // Hold still while nobody can see it.
  function setVisible(v) {
    if (v === visible) return
    visible = v
    if (!v) { clearTimeout(timer); hero.classList.remove('is-ticking') }
    else if (playing) { show(step); schedule() }
  }
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (es) {
      setVisible(es[0].isIntersecting && !document.hidden)
    }, { threshold: 0.25 }).observe(hero)
  }
  document.addEventListener('visibilitychange', function () {
    setVisible(!document.hidden)
  })

  hero.classList.add('is-live')
  if (reduce) { show(last); setPlaying(false) }
  else { show(0); setPlaying(true) }
})()

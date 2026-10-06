(function () {
  "use strict";

  var root = document.documentElement;
  var body = document.body;

  function setReadingProgress() {
    var progress = document.getElementById("readingProgress");
    if (!progress) return;

    var scrollable = root.scrollHeight - window.innerHeight;
    var value = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
    progress.style.width = Math.min(100, Math.max(0, value)) + "%";
  }

  setReadingProgress();
  window.addEventListener("scroll", setReadingProgress, { passive: true });
  window.addEventListener("resize", setReadingProgress);

  var menuToggle = document.getElementById("menuToggle");
  var siteNav = document.getElementById("siteNav");

  function closeMenu() {
    if (!menuToggle || !siteNav) return;
    menuToggle.setAttribute("aria-expanded", "false");
    siteNav.classList.remove("is-open");
    body.classList.remove("menu-open");
  }

  if (menuToggle && siteNav) {
    menuToggle.addEventListener("click", function () {
      var willOpen = menuToggle.getAttribute("aria-expanded") !== "true";
      menuToggle.setAttribute("aria-expanded", String(willOpen));
      siteNav.classList.toggle("is-open", willOpen);
      body.classList.toggle("menu-open", willOpen);
    });

    siteNav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", closeMenu);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeMenu();
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 920) closeMenu();
    });
  }

  var revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -7%", threshold: 0.08 });

    revealItems.forEach(function (item) {
      revealObserver.observe(item);
    });
  } else {
    revealItems.forEach(function (item) {
      item.classList.add("is-visible");
    });
  }

  var sectionLinks = Array.from(document.querySelectorAll('.site-nav a[href^="#"]'));
  var sections = sectionLinks
    .map(function (link) { return document.querySelector(link.getAttribute("href")); })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    var activeObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        sectionLinks.forEach(function (link) {
          link.classList.toggle("is-active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-35% 0px -55%", threshold: 0 });

    sections.forEach(function (section) {
      activeObserver.observe(section);
    });
  }

  var thresholdButtons = Array.from(document.querySelectorAll(".threshold-option"));
  var scoreInput = document.getElementById("scoreInput");
  var scoreValue = document.getElementById("scoreValue");
  var thresholdValue = document.getElementById("thresholdValue");
  var meterCard = document.querySelector(".meter-card");
  var decisionOutput = document.getElementById("decisionOutput");
  var activeThreshold = 0.0365;

  function updateDecision() {
    if (!scoreInput || !scoreValue || !thresholdValue || !meterCard || !decisionOutput) return;

    var score = Number(scoreInput.value);
    var maximum = Number(scoreInput.max);
    var certifies = score >= activeThreshold;
    var scorePosition = Math.min(100, Math.max(0, (score / maximum) * 100));
    var thresholdPosition = Math.min(100, Math.max(0, (activeThreshold / maximum) * 100));

    meterCard.style.setProperty("--score-position", scorePosition + "%");
    meterCard.style.setProperty("--threshold-position", thresholdPosition + "%");
    scoreValue.textContent = score.toFixed(4);
    thresholdValue.textContent = activeThreshold.toFixed(4);

    decisionOutput.classList.toggle("is-abstain", !certifies);
    decisionOutput.classList.toggle("is-certify", certifies);
    decisionOutput.querySelector(".decision-symbol").textContent = certifies ? "✓" : "↩";
    decisionOutput.querySelector("strong").textContent = certifies ? "Candidate for certification" : "Abstain";
    decisionOutput.querySelector("p").textContent = certifies
      ? "This score clears the active threshold. Final certification still requires clearing every tested generator."
      : "This generator can reproduce the query closely enough that synthetic provenance remains plausible.";
  }

  if (scoreInput) {
    scoreInput.addEventListener("input", updateDecision);
  }

  thresholdButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      activeThreshold = Number(button.dataset.threshold);
      thresholdButtons.forEach(function (item) {
        var isActive = item === button;
        item.classList.toggle("is-active", isActive);
        item.setAttribute("aria-pressed", String(isActive));
      });
      updateDecision();
    });
  });

  updateDecision();

  var tabs = Array.from(document.querySelectorAll('[role="tab"]'));

  function activateTab(tab, moveFocus) {
    tabs.forEach(function (item) {
      var selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;

      var panel = document.getElementById(item.getAttribute("aria-controls"));
      if (panel) panel.hidden = !selected;
    });

    if (moveFocus) tab.focus();
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () {
      activateTab(tab, false);
    });

    tab.addEventListener("keydown", function (event) {
      var nextIndex = index;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;

      if (nextIndex !== index) {
        event.preventDefault();
        activateTab(tabs[nextIndex], true);
      }
    });
  });

  var copyButton = document.getElementById("copyBibtex");
  var bibtex = document.getElementById("bibtex");

  function copyWithFallback(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy") ? resolve() : reject(new Error("Copy failed"));
      } catch (error) {
        reject(error);
      }
      area.remove();
    });
  }

  if (copyButton && bibtex) {
    copyButton.addEventListener("click", function () {
      var label = copyButton.querySelector("span");
      copyWithFallback(bibtex.textContent.trim()).then(function () {
        label.textContent = "Copied";
        copyButton.setAttribute("aria-label", "Citation copied");
        window.setTimeout(function () {
          label.textContent = "Copy";
          copyButton.removeAttribute("aria-label");
        }, 1600);
      }).catch(function () {
        label.textContent = "Select text";
        window.setTimeout(function () { label.textContent = "Copy"; }, 1600);
      });
    });
  }
})();

const header = document.querySelector("#site-header");
const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector("#nav-links");
const demo = document.querySelector("#method-demo");
const replayButton = document.querySelector("#replay-demo");

const updateHeader = () => header.classList.toggle("scrolled", window.scrollY > 16);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle.addEventListener("click", () => {
  const isOpen = navLinks.classList.toggle("open");
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

navLinks.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    navToggle.setAttribute("aria-expanded", "false");
  });
});

document.querySelector(".brand").addEventListener("click", () => {
  navLinks.classList.remove("open");
  navToggle.setAttribute("aria-expanded", "false");
});

replayButton.addEventListener("click", () => {
  demo.classList.remove("is-running");
  void demo.offsetWidth;
  demo.classList.add("is-running");
  replayButton.textContent = "Replaying…";
  window.setTimeout(() => {
    replayButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.7 6.3A9 9 0 1 0 21 12h-2a7 7 0 1 1-2-4.9L14 10h7V3l-2.3 3.3Z"/></svg> Replay';
  }, 3600);
});

const copyText = async (button) => {
  const target = document.getElementById(button.dataset.copyTarget);
  const text = target.innerText;
  try {
    await navigator.clipboard.writeText(text);
    const original = button.innerHTML;
    button.textContent = "Copied";
    window.setTimeout(() => { button.innerHTML = original; }, 1600);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(target);
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = "Selected";
  }
};

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => copyText(button));
});

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (reducedMotion || !("IntersectionObserver" in window)) {
  document.querySelectorAll(".reveal").forEach((item) => item.classList.add("visible"));
} else {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((item) => observer.observe(item));
}

document.querySelector("#footer-year").textContent = String(new Date().getFullYear());

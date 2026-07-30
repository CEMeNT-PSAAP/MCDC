document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll(".mcdc-theme-toggle");
  if (!buttons.length) {
    return;
  }

  const root = document.documentElement;

  const updateButtons = () => {
    const darkMode = root.dataset.theme === "dark";
    const nextMode = darkMode ? "light" : "dark";

    buttons.forEach((button) => {
      button.setAttribute("aria-label", `Switch to ${nextMode} mode`);
      button.setAttribute("title", `Switch to ${nextMode} mode`);
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const nextMode = root.dataset.theme === "dark" ? "light" : "dark";

      root.dataset.mode = nextMode;
      root.dataset.theme = nextMode;
      document.querySelectorAll(".dropdown-menu").forEach((menu) => {
        menu.classList.toggle("dropdown-menu-dark", nextMode === "dark");
      });

      localStorage.setItem("mode", nextMode);
      localStorage.setItem("theme", nextMode);
      updateButtons();
    });
  });

  new MutationObserver(updateButtons).observe(root, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  updateButtons();
});

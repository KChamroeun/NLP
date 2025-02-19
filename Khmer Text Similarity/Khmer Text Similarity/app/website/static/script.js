function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');

    if (sidebar.style.left === '0px') {
        sidebar.style.left = '-250px'; 
        sidebarToggle.style.opacity = '1'; 
    } else {
        sidebar.style.left = '0px'; 
        sidebarToggle.style.opacity = '0'; 
    }
}


document.addEventListener('click', function(event) {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');

    if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
        if (sidebar.style.left === '0px') {
            sidebar.style.left = '-250px'; 
            sidebarToggle.style.opacity = '1'; 
        }
    }
});
  
document.addEventListener('DOMContentLoaded', function() {
    const dropdownItems1 = document.querySelectorAll('.dropdown-item1');
    const dropdownLabel1 = document.getElementById('dropdown-label1');
  
    dropdownItems1.forEach(item => {
      item.addEventListener('click', function(event) {
        event.preventDefault();
        dropdownLabel1.textContent = this.textContent;
        document.getElementById('touch1').checked = false;
      });
    });
  
    const dropdownItems2 = document.querySelectorAll('.dropdown-item2');
    const dropdownLabel2 = document.getElementById('dropdown-label2');
  
    dropdownItems2.forEach(item => {
      item.addEventListener('click', function(event) {
        event.preventDefault();
        dropdownLabel2.textContent = this.textContent;
        document.getElementById('touch2').checked = false;
      });
    });
  });
  
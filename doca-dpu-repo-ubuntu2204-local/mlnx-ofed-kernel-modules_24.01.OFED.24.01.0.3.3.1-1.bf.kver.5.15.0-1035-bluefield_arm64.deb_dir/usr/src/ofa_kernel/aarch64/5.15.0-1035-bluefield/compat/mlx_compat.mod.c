#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x87dbb23e, "module_layout" },
	{ 0x2d3385d3, "system_wq" },
	{ 0xb43f9365, "ktime_get" },
	{ 0xcc4eda8, "param_ops_charp" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x8da6585d, "__stack_chk_fail" },
	{ 0x92997ed8, "_printk" },
	{ 0xc5b6f236, "queue_work_on" },
	{ 0x2b915b68, "pcie_capability_read_word" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "313B256FB318D7B6B62DDCB");
